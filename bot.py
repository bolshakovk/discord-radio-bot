from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from urllib.parse import urlparse

import discord
from discord.ext import commands, tasks
from discord.ext import voice_recv
import numpy as np
from faster_whisper import WhisperModel
from thefuzz import fuzz



# ============================================================
# CONFIG
# ============================================================

# IMPORTANT: do not put the Discord token into this file.
# PowerShell example:
#   $env:DISCORD_TOKEN = "YOUR_NEW_TOKEN"
DISCORD_TOKEN = "сюда писать токен"

COMMAND_CHANNEL_ID = 434827232532496385
MUSIC_CHANNEL_ID = 1288596749274976348

FFMPEG_PATH = (
    r"C:\vibecoding\discord-radio-bot"
    r"\ffmpeg-master-latest-win64-gpl"
    r"\ffmpeg-master-latest-win64-gpl"
    r"\bin\ffmpeg.exe"
)

WHISPER_MODEL_SIZE = "small"
VOICE_PREFIX = "вонючка"

# Voice phrase detection
VOICE_SILENCE_SECONDS = 0.55
MIN_VOICE_SECONDS = 0.25
MAX_VOICE_SECONDS = 12.0
MAX_VOICE_QUEUE = 10

# Do not repeat one of the last N tracks unless the collection is exhausted.
NO_REPEAT_LAST_N = 20

# Refresh Discord attachment URLs periodically.
# This is useful because Discord attachment URLs can be signed URLs.
CACHE_REFRESH_MINUTES = 10

# Maximum time we are willing to wait for an old FFmpeg process to die.
FFMPEG_SHUTDOWN_TIMEOUT = 8.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("radio")

# Reduce harmless voice-recv noise while keeping warnings/errors.
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)


# ============================================================
# DISCORD
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
)


# ============================================================
# WHISPER
# ============================================================

whisper_model: WhisperModel | None = None
whisper_model_lock = threading.Lock()


def load_whisper_model() -> WhisperModel:
    global whisper_model

    if whisper_model is not None:
        return whisper_model

    with whisper_model_lock:
        if whisper_model is not None:
            return whisper_model

        log.info(
            "[VOICE] 🧠 Загружаю локальную Whisper-модель: %s",
            WHISPER_MODEL_SIZE,
        )

        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )

        log.info("[VOICE] 🧠 Whisper готов")

    return whisper_model


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("ё", "е")

    text = re.sub(
        r"[^\w\s-]+",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_title(filename: str) -> str:
    title = os.path.splitext(filename)[0]
    title = re.sub(r"[_\-.]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title or filename


def filename_from_url(url: str) -> str | None:
    try:
        path = urlparse(url).path
        filename = path.rsplit("/", 1)[-1]
        if filename.lower().endswith(".mp3"):
            return filename
    except Exception:
        pass
    return None


# ============================================================
# VOICE COMMAND SINK
# ============================================================


class VoiceCommandSink(voice_recv.AudioSink):
    """
    Receives decoded PCM from discord-ext-voice-recv, groups speech
    into phrases by silence, then sends audio to Whisper.
    """

    SAMPLE_RATE = 48000
    CHANNELS = 2
    SAMPLE_WIDTH = 2

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        handle_voice_command,
    ) -> None:
        super().__init__()

        self.loop = loop
        self.handle_voice_command = handle_voice_command

        self.buffers: dict[int, bytearray] = {}
        self.last_packet_time: dict[int, float] = {}
        self.users: dict[int, object] = {}

        self.state_lock = threading.Lock()

        self.queue: asyncio.Queue[tuple[object, bytes]] = asyncio.Queue(
            maxsize=MAX_VOICE_QUEUE
        )

        self.worker_task: asyncio.Task | None = None
        self.monitor_task: asyncio.Task | None = None
        self.closed = False

    def wants_opus(self) -> bool:
        # We want decoded PCM.
        return False

    def start_worker(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = self.loop.create_task(self.worker())

        if self.monitor_task is None or self.monitor_task.done():
            self.monitor_task = self.loop.create_task(self.monitor_speech())

        log.info("[VOICE] 🎙️ Whisper worker запущен")

    def cleanup(self) -> None:
        self.closed = True

        with self.state_lock:
            self.buffers.clear()
            self.last_packet_time.clear()
            self.users.clear()

        for task in (self.worker_task, self.monitor_task):
            if task is not None and not task.done():
                task.cancel()

    def write(self, user, data) -> None:
        if self.closed or user is None or data is None:
            return

        # Ignore bot users. This prevents accidental recognition of bot audio.
        if getattr(user, "bot", False):
            return

        pcm = getattr(data, "pcm", None)
        if not pcm:
            return

        user_id = user.id
        now = time.monotonic()

        with self.state_lock:
            buffer = self.buffers.setdefault(user_id, bytearray())
            buffer.extend(pcm)
            self.last_packet_time[user_id] = now
            self.users[user_id] = user

            max_bytes = int(
                self.SAMPLE_RATE
                * self.CHANNELS
                * self.SAMPLE_WIDTH
                * MAX_VOICE_SECONDS
            )

            if len(buffer) >= max_bytes:
                raw_audio = bytes(buffer)
                self.buffers[user_id] = bytearray()
                self.last_packet_time.pop(user_id, None)
            else:
                raw_audio = None

        if raw_audio:
            self._schedule_enqueue(user, raw_audio)

    def _schedule_enqueue(self, member, raw_audio: bytes) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                self.enqueue_audio(member, raw_audio),
                self.loop,
            )
        except RuntimeError:
            log.exception("[VOICE] ❌ Не удалось поставить аудио в очередь")

    async def monitor_speech(self) -> None:
        while True:
            try:
                await asyncio.sleep(0.10)
                now = time.monotonic()
                finalized: list[tuple[object, bytes]] = []

                with self.state_lock:
                    for user_id, last_time in list(self.last_packet_time.items()):
                        if now - last_time < VOICE_SILENCE_SECONDS:
                            continue

                        raw_audio = bytes(
                            self.buffers.get(user_id, bytearray())
                        )
                        user = self.users.get(user_id)

                        self.buffers[user_id] = bytearray()
                        self.last_packet_time.pop(user_id, None)

                        if user is not None and raw_audio:
                            finalized.append((user, raw_audio))

                for user, raw_audio in finalized:
                    await self.enqueue_audio(user, raw_audio)

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("[VOICE] ❌ Ошибка monitor_speech")

    async def enqueue_audio(self, member, raw_audio: bytes) -> None:
        bytes_per_second = (
            self.SAMPLE_RATE
            * self.CHANNELS
            * self.SAMPLE_WIDTH
        )

        duration = len(raw_audio) / bytes_per_second

        if duration < MIN_VOICE_SECONDS:
            return

        if self.queue.full():
            log.warning(
                "[VOICE] ⚠️ Whisper queue заполнена, фраза отброшена"
            )
            return

        log.info(
            "[VOICE] 🎙️ Получена фраза от %s: %.2fs",
            member.display_name,
            duration,
        )

        await self.queue.put((member, raw_audio))

    async def worker(self) -> None:
        while True:
            try:
                member, raw_audio = await self.queue.get()

                try:
                    text = await self.loop.run_in_executor(
                        None,
                        self.transcribe,
                        raw_audio,
                    )

                    if not text:
                        continue

                    log.info(
                        '[VOICE] 🗣️ %s: "%s"',
                        member.display_name,
                        text,
                    )

                    await self.handle_voice_command(
                        member,
                        text,
                    )

                except Exception:
                    log.exception(
                        "[VOICE] ❌ Ошибка обработки голосовой команды"
                    )
                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("[VOICE] ❌ Ошибка Voice worker")

    @staticmethod
    def transcribe(raw_audio: bytes) -> str:
        model = load_whisper_model()

        pcm = np.frombuffer(
            raw_audio,
            dtype=np.int16,
        )

        if pcm.size == 0:
            return ""

        # 48 kHz stereo -> mono
        if pcm.size % 2:
            pcm = pcm[:-1]

        pcm = pcm.reshape(-1, 2)
        audio = pcm.mean(axis=1).astype(np.float32)
        audio /= 32768.0

        # 48 kHz -> 16 kHz using simple 3:1 decimation.
        usable_size = (len(audio) // 3) * 3
        if usable_size == 0:
            return ""

        audio = audio[:usable_size]
        audio = audio.reshape(-1, 3).mean(axis=1)

        if audio.size == 0:
            return ""

        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak

        duration = audio.size / 16000.0
        log.info(
            "[VOICE] 🧠 Whisper audio: %.2fs / 16kHz",
            duration,
        )

        try:
            segments, _info = model.transcribe(
                audio,
                language="ru",
                beam_size=5,
                best_of=5,
                temperature=0,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=(
                    "Вонючка. "
                    "Стоп. "
                    "Играй. "
                    "Скип. "
                    "Включи. "
                    "Поставь. "
                    "Запусти. "
                    "Следующий трек. "
                    "Дальше. "
                    "Радио."
                ),
            )

            return " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ).strip()

        except Exception:
            log.exception("[VOICE] ❌ Ошибка Whisper")
            return ""


# ============================================================
# RADIO MANAGER
# ============================================================


class RadioManager:
    def __init__(self) -> None:
        self.bot = bot

        self.music_channel: discord.TextChannel | None = None
        self.tracks: list[dict] = []
        self.current_track: dict | None = None
        self.history: deque[str] = deque(maxlen=NO_REPEAT_LAST_N)

        self.voice_client: voice_recv.VoiceRecvClient | None = None
        self.voice_sink: VoiceCommandSink | None = None
        self.voice_channel: discord.VoiceChannel | None = None

        # Logical radio state. This stays True while radio should continue.
        self.is_playing = False

        # Changes whenever a new playback request invalidates old callbacks.
        self._playback_generation = 0

        # All playback transitions are serialized here.
        self._playback_lock = asyncio.Lock()

        # Event loop used by FFmpeg callbacks and voice receive.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Search results are isolated per user.
        self._search_results: dict[int, list[tuple[int, dict]]] = {}

        self._scan_lock = asyncio.Lock()

        # Listener generation prevents an intentional listener stop from
        # being immediately auto-restarted by the listener callback.
        self._listener_generation = 0

    # ----------------------------------------------------------
    # COLLECTION
    # ----------------------------------------------------------

    async def scan_collection(self) -> int:
        async with self._scan_lock:
            channel = self.bot.get_channel(MUSIC_CHANNEL_ID)

            if channel is None:
                log.error(
                    "[RADIO] ❌ Музыкальный канал %s не найден",
                    MUSIC_CHANNEL_ID,
                )
                return 0

            self.music_channel = channel
            tracks: list[dict] = []
            seen_urls: set[str] = set()

            log.info(
                "[RADIO] 🔍 Сканирую музыкальный канал: #%s",
                getattr(channel, "name", MUSIC_CHANNEL_ID),
            )

            try:
                async for message in channel.history(
                    limit=None,
                    oldest_first=True,
                ):
                    for attachment in message.attachments:
                        if not attachment.filename.lower().endswith(".mp3"):
                            continue

                        base_url = attachment.url.split("?", 1)[0]
                        if base_url in seen_urls:
                            continue

                        seen_urls.add(base_url)
                        tracks.append(
                            {
                                "title": clean_title(attachment.filename),
                                "url": attachment.url,
                                "message_id": message.id,
                            }
                        )

                    if message.content:
                        urls = re.findall(
                            r"https?://\S+",
                            message.content,
                            flags=re.IGNORECASE,
                        )

                        for raw_url in urls:
                            url = raw_url.strip("<>()[]")
                            filename = filename_from_url(url)

                            if filename is None:
                                continue

                            base_url = url.split("?", 1)[0]
                            if base_url in seen_urls:
                                continue

                            seen_urls.add(base_url)
                            tracks.append(
                                {
                                    "title": clean_title(filename),
                                    "url": url,
                                    "message_id": message.id,
                                }
                            )

            except Exception:
                log.exception(
                    "[RADIO] ❌ Ошибка сканирования музыкального канала"
                )
                return 0

            self.tracks = tracks

            log.info(
                "[RADIO] ✅ Найдено треков: %d",
                len(self.tracks),
            )

            return len(self.tracks)

    # ----------------------------------------------------------

    def find_track(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[int, dict]]:
        if not self.tracks:
            return []

        query_normalized = normalize_text(query)
        if not query_normalized:
            return []

        results: list[tuple[int, dict]] = []

        for track in self.tracks:
            title = normalize_text(track["title"])

            score_partial = fuzz.partial_ratio(
                query_normalized,
                title,
            )
            score_token = fuzz.token_set_ratio(
                query_normalized,
                title,
            )
            score_ratio = fuzz.ratio(
                query_normalized,
                title,
            )

            score = max(
                score_partial,
                score_token,
                score_ratio,
            )

            results.append((score, track))

        results.sort(key=lambda item: item[0], reverse=True)
        return results[:limit]

    # ----------------------------------------------------------
    # TRACK CHOICE
    # ----------------------------------------------------------

    def choose_random_track(self) -> dict | None:
        if not self.tracks:
            return None

        available = [
            track
            for track in self.tracks
            if track["url"].split("?", 1)[0] not in {
                url.split("?", 1)[0] for url in self.history
            }
        ]

        if not available:
            log.info(
                "[RADIO] 🔄 Все треки были недавно сыграны, сбрасываю историю"
            )

            current_base = (
                self.current_track["url"].split("?", 1)[0]
                if self.current_track
                else None
            )

            self.history.clear()

            if current_base:
                self.history.append(current_base)

            available = [
                track
                for track in self.tracks
                if track["url"].split("?", 1)[0] != current_base
            ]

        if not available:
            available = self.tracks

        return random.choice(available)

    # ----------------------------------------------------------
    # FFmpeg HELPERS
    # ----------------------------------------------------------

    def create_source(self, url: str) -> discord.AudioSource:
        return discord.FFmpegPCMAudio(
            url,
            executable=FFMPEG_PATH,
            before_options=(
                "-reconnect 1 "
                "-reconnect_streamed 1 "
                "-reconnect_delay_max 5"
            ),
            options="-vn",
        )

    @staticmethod
    def get_source_process(source) :
        process = getattr(source, "_process", None)
        if process is None:
            return None

        if callable(getattr(process, "poll", None)):
            return process

        # FFmpegPCMAudio.cleanup() replaces _process with a sentinel object.
        return None

    async def wait_for_source_shutdown(
        self,
        source,
        timeout: float = FFMPEG_SHUTDOWN_TIMEOUT,
    ) -> None:
        """
        IMPORTANT:
        VoiceClient.stop() makes is_playing() false immediately by setting
        _player=None, but the AudioPlayer thread still has to finish and
        FFmpegPCMAudio.cleanup() happens after the after-callback.

        Therefore we wait for the actual FFmpeg process here instead of
        relying on VoiceClient.is_playing().
        """
        if source is None:
            await asyncio.sleep(0.05)
            return

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            process = self.get_source_process(source)

            if process is None:
                return

            try:
                if process.poll() is not None:
                    return
            except Exception:
                return

            await asyncio.sleep(0.05)

        log.warning(
            "[RADIO] ⚠️ FFmpeg не завершился за %.1fs — продолжаю",
            timeout,
        )

    def stop_playback_only(self) -> None:
        """Stops sending audio without stopping VoiceRecvClient listening."""
        if self.voice_client is None:
            return

        stop_playing = getattr(self.voice_client, "stop_playing", None)

        if callable(stop_playing):
            stop_playing()
        else:
            # Fallback for a plain discord.py VoiceClient.
            self.voice_client.stop()

    # ----------------------------------------------------------
    # VOICE CONNECTION
    # ----------------------------------------------------------

    async def connect_voice(
        self,
        channel: discord.VoiceChannel,
    ):
        existing = self.voice_client

        if existing is not None:
            try:
                if (
                    existing.is_connected()
                    and existing.channel is not None
                    and existing.channel.id == channel.id
                    and isinstance(existing, voice_recv.VoiceRecvClient)
                ):
                    self.voice_channel = channel
                    return existing
            except Exception:
                pass

            try:
                await existing.disconnect()
            except Exception:
                log.exception("[VOICE] ❌ Ошибка отключения старого voice client")

            self.voice_client = None
            self.voice_sink = None

        log.info(
            "[VOICE] 🎙️ Подключаюсь к: %s",
            channel.name,
        )

        vc = await channel.connect(
            cls=voice_recv.VoiceRecvClient,
        )

        self.voice_client = vc
        self.voice_channel = channel

        log.info("[VOICE] ✅ VoiceRecvClient подключён")
        return vc

    async def start_voice_listener(
        self,
        channel: discord.VoiceChannel,
    ) -> None:
        await self.connect_voice(channel)

        if self.voice_client is None:
            raise RuntimeError("VoiceClient отсутствует после подключения")

        # If listener is already alive, do not touch it.
        try:
            if self.voice_client.is_listening():
                log.info("[VOICE] 🎧 Listener уже работает")
                return
        except Exception:
            pass

        old_sink = self.voice_sink
        if old_sink is not None:
            try:
                self.voice_client.stop_listening()
            except Exception:
                pass

            # Give the reader thread a moment to detach from the old sink.
            for _ in range(20):
                try:
                    if not self.voice_client.is_listening():
                        break
                except Exception:
                    break
                await asyncio.sleep(0.05)

            try:
                old_sink.cleanup()
            except Exception:
                pass

            self.voice_sink = None

        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        self._listener_generation += 1
        listener_generation = self._listener_generation

        sink = VoiceCommandSink(
            self._loop,
            self.handle_voice_command,
        )

        self.voice_sink = sink
        sink.start_worker()

        def after_listener(error):
            asyncio.run_coroutine_threadsafe(
                self.on_listener_finished(
                    listener_generation,
                    error,
                ),
                self._loop,
            )

        self.voice_client.listen(
            sink,
            after=after_listener,
        )

        log.info("[VOICE] 🎧 Начал слушать голос")

    async def on_listener_finished(
        self,
        generation: int,
        error: Exception | None,
    ) -> None:
        if error:
            log.error(
                "[VOICE] ❌ Listener завершился с ошибкой: %s",
                error,
            )
        else:
            log.warning("[VOICE] ⚠️ Listener завершился")

        if generation != self._listener_generation:
            return

        if self.voice_client is None:
            return

        try:
            if not self.voice_client.is_connected():
                return
        except Exception:
            return

        await asyncio.sleep(0.5)

        if generation != self._listener_generation:
            return

        if self.voice_channel is not None:
            try:
                if not self.voice_client.is_listening():
                    log.info("[VOICE] 🔁 Перезапускаю voice listener")
                    await self.start_voice_listener(self.voice_channel)
            except Exception:
                log.exception("[VOICE] ❌ Не удалось перезапустить listener")

    async def stop_voice_listener(self) -> None:
        self._listener_generation += 1

        if self.voice_client is None:
            self.voice_sink = None
            return

        try:
            self.voice_client.stop_listening()
        except Exception:
            pass

        sink = self.voice_sink
        self.voice_sink = None

        if sink is not None:
            try:
                sink.cleanup()
            except Exception:
                pass

    async def ensure_listener(
        self,
        channel: discord.VoiceChannel,
    ) -> None:
        if (
            self.voice_client is None
            or not self.voice_client.is_connected()
        ):
            await self.start_voice_listener(channel)
            return

        try:
            if (
                self.voice_client.channel is None
                or self.voice_client.channel.id != channel.id
            ):
                await self.start_voice_listener(channel)
                return
        except Exception:
            await self.start_voice_listener(channel)
            return

        try:
            if not self.voice_client.is_listening():
                await self.start_voice_listener(channel)
        except Exception:
            await self.start_voice_listener(channel)

    # ----------------------------------------------------------
    # PLAYBACK
    # ----------------------------------------------------------

    async def _play_track_unlocked(
        self,
        track: dict,
        generation: int,
    ) -> bool:
        if self.voice_client is None:
            log.warning("[RADIO] ❌ Нет голосового подключения")
            return False

        if generation != self._playback_generation:
            log.info(
                "[RADIO] ℹ️ Не запускаю устаревший generation=%s",
                generation,
            )
            return False

        try:
            if self.voice_client.is_playing():
                log.warning(
                    "[RADIO] ⚠️ Перед новым треком ещё что-то играет; останавливаю"
                )
                old_source = self.voice_client.source
                self.stop_playback_only()
                await self.wait_for_source_shutdown(old_source)
        except Exception:
            log.exception("[RADIO] ❌ Ошибка проверки старого player")

        if generation != self._playback_generation:
            return False

        source = None

        try:
            source = self.create_source(track["url"])

            self.current_track = track
            self.is_playing = True

            base_url = track["url"].split("?", 1)[0]
            self.history.append(base_url)

            log.info(
                "[RADIO] ▶️ Запускаю: %s",
                track["title"],
            )
            log.info(
                "[RADIO] URL: %s",
                track["url"],
            )

            def after_play(error):
                if self._loop is None:
                    return

                if error:
                    log.error(
                        "[RADIO] ❌ Ошибка воспроизведения '%s': %s",
                        track["title"],
                        error,
                    )

                try:
                    asyncio.run_coroutine_threadsafe(
                        self.on_track_finished(
                            generation,
                            track,
                            source,
                        ),
                        self._loop,
                    )
                except RuntimeError:
                    log.exception(
                        "[RADIO] ❌ Не удалось передать after callback в event loop"
                    )

            self.voice_client.play(
                source,
                after=after_play,
            )

            log.info(
                "[RADIO] 🎵 Сейчас играет: %s",
                track["title"],
            )
            return True

        except Exception:
            log.exception(
                "[RADIO] ❌ Не удалось запустить FFmpeg: %s",
                track["title"],
            )

            self.current_track = None
            self.is_playing = False

            if source is not None:
                try:
                    source.cleanup()
                except Exception:
                    pass

            return False

    async def on_track_finished(
        self,
        generation: int,
        track: dict,
        source,
    ) -> None:
        """
        Called by discord.py after the AudioPlayer reaches its end.
        We still wait for FFmpeg/source cleanup before starting another track.
        """
        async with self._playback_lock:
            if generation != self._playback_generation:
                log.info(
                    "[RADIO] ℹ️ Игнорирую callback старого трека: %s",
                    track["title"],
                )
                return

            if not self.is_playing:
                log.info(
                    "[RADIO] ℹ️ Радио уже остановлено: %s",
                    track["title"],
                )
                return

            log.info(
                "[RADIO] ⏹️ Трек закончился: %s",
                track["title"],
            )

            await self.wait_for_source_shutdown(source)

            # A manual command could have arrived while we waited.
            if generation != self._playback_generation:
                log.info(
                    "[RADIO] ℹ️ Новый playback request пришёл во время завершения трека"
                )
                return

            if not self.is_playing:
                return

            next_track = self.choose_random_track()
            if next_track is None:
                self.is_playing = False
                self.current_track = None
                log.error("[RADIO] ❌ Не удалось выбрать следующий трек")
                return

            await self._play_track_unlocked(
                next_track,
                generation,
            )

    async def play_next(self) -> bool:
        if not self.tracks:
            await self.scan_collection()

        if not self.tracks:
            log.warning("[RADIO] ❌ Нет доступных треков")
            return False

        async with self._playback_lock:
            generation = self._playback_generation
            track = self.choose_random_track()

            if track is None:
                return False

            return await self._play_track_unlocked(
                track,
                generation,
            )

    async def start_radio(
        self,
        channel: discord.VoiceChannel,
    ) -> bool:
        await self.ensure_listener(channel)

        if not self.tracks:
            await self.scan_collection()

        if not self.tracks:
            return False

        async with self._playback_lock:
            # Already playing: nothing to restart.
            try:
                if self.voice_client is not None and self.voice_client.is_playing():
                    self.is_playing = True
                    return True
            except Exception:
                pass

            self._playback_generation += 1
            generation = self._playback_generation
            self.is_playing = True

            track = self.choose_random_track()
            if track is None:
                self.is_playing = False
                return False

            return await self._play_track_unlocked(
                track,
                generation,
            )

    async def stop_radio(
        self,
        disconnect: bool = False,
    ) -> None:
        async with self._playback_lock:
            self.is_playing = False
            self.current_track = None
            self._playback_generation += 1

            old_source = None

            if self.voice_client is not None:
                try:
                    old_source = self.voice_client.source
                except Exception:
                    old_source = None

                try:
                    if self.voice_client.is_playing():
                        self.stop_playback_only()
                except Exception:
                    log.exception("[RADIO] ❌ Ошибка остановки playback")

            await self.wait_for_source_shutdown(old_source)

            if disconnect:
                await self.stop_voice_listener()

                if self.voice_client is not None:
                    try:
                        await self.voice_client.disconnect()
                    except Exception:
                        log.exception("[VOICE] ❌ Ошибка отключения")

                    self.voice_client = None
                    self.voice_channel = None

            log.info(
                "[RADIO] ⏹️ Радио остановлено; listener=%s",
                not disconnect,
            )

    async def skip_track(self) -> bool:
        async with self._playback_lock:
            if self.voice_client is None:
                log.warning("[RADIO] ❌ Нет голосового подключения")
                return False

            if not self.is_playing:
                log.info("[RADIO] ℹ️ Skip проигнорирован: радио остановлено")
                return False

            self._playback_generation += 1
            generation = self._playback_generation
            self.is_playing = True

            old_track = self.current_track
            old_source = None

            try:
                if self.voice_client.is_playing():
                    old_source = self.voice_client.source
                    self.stop_playback_only()
            except Exception:
                log.exception("[RADIO] ❌ Ошибка остановки текущего трека")

            log.info(
                "[RADIO] ⏭️ SKIP: %s",
                old_track["title"] if old_track else "нет текущего трека",
            )

            # THIS is the important part: wait for the actual FFmpeg process,
            # not VoiceClient.is_playing().
            await self.wait_for_source_shutdown(old_source)

            if generation != self._playback_generation:
                return False

            next_track = self.choose_random_track()
            if next_track is None:
                self.is_playing = False
                self.current_track = None
                return False

            ok = await self._play_track_unlocked(
                next_track,
                generation,
            )

            if ok:
                log.info(
                    "[RADIO] ✅ SKIP завершён → %s",
                    next_track["title"],
                )
            return ok

    async def play_specific(
        self,
        query: str,
    ) -> tuple[dict, int] | None:
        results = self.find_track(query, limit=5)
        if not results:
            return None

        score, track = results[0]

        async with self._playback_lock:
            self._playback_generation += 1
            generation = self._playback_generation
            self.is_playing = True

            old_source = None

            try:
                if self.voice_client is not None and self.voice_client.is_playing():
                    old_source = self.voice_client.source
                    self.stop_playback_only()
            except Exception:
                log.exception("[RADIO] ❌ Ошибка остановки текущего трека")

            await self.wait_for_source_shutdown(old_source)

            if generation != self._playback_generation:
                return None

            ok = await self._play_track_unlocked(
                track,
                generation,
            )

            if not ok:
                return None

            return track, score

    # ----------------------------------------------------------
    # VOICE COMMAND PARSING
    # ----------------------------------------------------------

    @staticmethod
    def normalize_voice_text(text: str) -> str:
        return normalize_text(text)

    def extract_voice_command(self, text: str) -> str | None:
        text = self.normalize_voice_text(text)
        if not text:
            return None

        # Normal recognition.
        match = re.search(r"\bвонючка\b", text)
        if match:
            return text[match.end():].strip()

        # Fallback for minor Whisper mistakes in the wake word.
        words = text.split()
        if not words:
            return None

        similarity = SequenceMatcher(
            None,
            words[0],
            VOICE_PREFIX,
        ).ratio()

        if similarity >= 0.72:
            return " ".join(words[1:]).strip()

        return None

    def parse_voice_command(
        self,
        command: str,
    ) -> tuple[str, str | None] | None:
        command = self.normalize_voice_text(command)
        if not command:
            return None

        # Longer/specific patterns first.
        track_match = re.search(
            r"\b(?:включи|поставь|запусти)\b\s+(.+)",
            command,
        )
        if track_match:
            return "track", track_match.group(1).strip()

        if re.search(
            r"\b(?:стоп|стопни|остановись|останови|остановить|стопа|стопп|топ)\b",
            command,
        ):
            return "stop", None

        if re.search(
            r"\b(?:играй|включай|продолжай|запускай)\b",
            command,
        ):
            return "play", None

        if re.search(
            r"\b(?:скип|скипни|следующий|дальше)\b",
            command,
        ):
            return "skip", None

        return None

    async def handle_voice_command(
        self,
        member,
        text: str,
    ) -> None:
        log.info(
            '[VOICE] 🔥 HANDLE: %s: "%s"',
            member.display_name,
            text,
        )

        if self.voice_client is None:
            log.info("[VOICE] ❌ Нет voice client")
            return

        # Only users in the same voice channel can control the radio.
        try:
            if (
                member.voice is None
                or self.voice_client.channel is None
                or member.voice.channel.id != self.voice_client.channel.id
            ):
                log.info(
                    "[VOICE] 🚫 Игнор: пользователь не в канале бота"
                )
                return
        except Exception:
            log.exception("[VOICE] ❌ Не удалось проверить voice state")
            return

        command = self.extract_voice_command(text)

        log.info(
            '[VOICE] 🔎 Извлечённая команда: "%s"',
            command,
        )

        if command is None:
            return

        if not command:
            log.info("[VOICE] ❓ После wake word нет команды")
            return

        parsed = self.parse_voice_command(command)

        log.info("[VOICE] 🧩 Parsed: %s", parsed)

        if parsed is None:
            log.info(
                '[VOICE] ❓ Неизвестная команда: "%s"',
                command,
            )
            return

        command_type, argument = parsed

        if command_type == "stop":
            log.info("[VOICE] ⏹️ Выполняю STOP")
            await self.stop_radio(disconnect=False)
            log.info(
                "[VOICE] ✅ Радио остановлено, остаюсь слушать канал"
            )
            return

        if command_type == "play":
            if self.voice_client.channel is not None:
                ok = await self.start_radio(self.voice_client.channel)
                log.info(
                    "[VOICE] ▶️ PLAY завершён: ok=%s",
                    ok,
                )
            return

        if command_type == "skip":
            log.info("[VOICE] ⏭️ Выполняю SKIP")
            ok = await self.skip_track()
            log.info(
                "[VOICE] ✅ SKIP завершён: ok=%s",
                ok,
            )
            return

        if command_type == "track":
            if not argument:
                return

            log.info(
                '[VOICE] 🎵 Ищу трек: "%s"',
                argument,
            )

            result = await self.play_specific(argument)
            if result is None:
                log.info(
                    '[VOICE] ❌ Не найден/не запущен трек: "%s"',
                    argument,
                )
                return

            track, score = result
            log.info(
                '[VOICE] ✅ Включаю "%s" (score=%d)',
                track["title"],
                score,
            )
            return

    # ----------------------------------------------------------
    # TEXT COMMANDS
    # ----------------------------------------------------------

    async def command_listen(self, ctx: commands.Context) -> None:
        if ctx.author.voice is None or ctx.author.voice.channel is None:
            await ctx.send("❌ Сначала зайди в голосовой канал!")
            return

        channel = ctx.author.voice.channel

        ok = await self.start_radio(channel)
        if ok:
            await ctx.send(
                f"🎙️ Слушаю **{channel.name}** и запускаю радио."
            )
        else:
            await ctx.send("❌ Не удалось запустить радио.")

    async def command_stop(self, ctx: commands.Context) -> None:
        await self.stop_radio(disconnect=True)
        await ctx.send("⏹️ Радио остановлено, отключился от голосового канала.")

    async def command_skip(self, ctx: commands.Context) -> None:
        ok = await self.skip_track()
        if ok:
            await ctx.send("⏭️ Следующий трек запущен.")
        else:
            await ctx.send("❌ Радио сейчас не играет.")

    async def command_find(
        self,
        ctx: commands.Context,
        query: str,
    ) -> None:
        if not self.tracks:
            await self.scan_collection()

        results = self.find_track(query, limit=5)

        if not results:
            await ctx.send(
                f'❌ Ничего не найдено по запросу **"{query}"**.'
            )
            return

        self._search_results[ctx.author.id] = results

        lines = [
            f'🔎 Результаты по запросу **"{query}"**:',
        ]

        for index, (score, track) in enumerate(results, start=1):
            lines.append(
                f"`{index}.` **{track['title']}** — `{score}%`"
            )

        lines.append(
            "\nИспользуй `!включи <номер>` чтобы включить найденный трек."
        )

        await ctx.send("\n".join(lines))

    async def command_play(
        self,
        ctx: commands.Context,
        number: int,
    ) -> None:
        results = self._search_results.get(ctx.author.id)

        if not results:
            await ctx.send(
                "❌ Сначала используй `!найди <название>`."
            )
            return

        if number < 1 or number > len(results):
            await ctx.send(
                f"❌ Укажи номер от 1 до {len(results)}."
            )
            return

        if ctx.author.voice is None or ctx.author.voice.channel is None:
            await ctx.send("❌ Сначала зайди в голосовой канал!")
            return

        channel = ctx.author.voice.channel
        await self.ensure_listener(channel)

        _, track = results[number - 1]

        # Play exact search result via the same safe transition logic.
        async with self._playback_lock:
            self._playback_generation += 1
            generation = self._playback_generation
            self.is_playing = True

            old_source = None
            try:
                if self.voice_client is not None and self.voice_client.is_playing():
                    old_source = self.voice_client.source
                    self.stop_playback_only()
            except Exception:
                log.exception("[RADIO] ❌ Ошибка остановки текущего трека")

            await self.wait_for_source_shutdown(old_source)

            ok = await self._play_track_unlocked(
                track,
                generation,
            )

        if ok:
            await ctx.send(
                f"🎵 Включаю: **{track['title']}**"
            )
        else:
            await ctx.send("❌ Не удалось запустить трек.")

    async def command_queue(self, ctx: commands.Context) -> None:
        if not self.tracks:
            await self.scan_collection()

        if not self.tracks:
            await ctx.send("📭 Музыкальная коллекция пуста.")
            return

        lines = ["📻 **Вонючка**"]

        if self.current_track:
            lines.append(
                f"\n▶️ Сейчас: **{self.current_track['title']}**"
            )
        else:
            lines.append("\n⏹️ Сейчас ничего не играет.")

        if self.history:
            lines.append("\n📜 Последние треки:")

            history_bases = list(self.history)[-10:][::-1]
            for index, base_url in enumerate(history_bases, start=1):
                track = next(
                    (
                        item
                        for item in self.tracks
                        if item["url"].split("?", 1)[0] == base_url
                    ),
                    None,
                )
                if track:
                    marker = "▶️" if index == 1 and self.current_track else "  "
                    lines.append(
                        f"{marker} `{index}.` {track['title']}"
                    )

        lines.append(
            f"\n📊 Всего MP3: **{len(self.tracks)}**"
        )

        await ctx.send("\n".join(lines))

    async def command_update(self, ctx: commands.Context) -> None:
        await ctx.send("🔄 Обновляю коллекцию...")
        count = await self.scan_collection()
        await ctx.send(
            f"✅ Готово. Найдено **{count}** MP3."
        )

    async def command_help(self, ctx: commands.Context) -> None:
        await ctx.send(
            "**🎵 Вонючка**\n\n"
            "`!слушай` — подключиться и запустить радио\n"
            "`!стоп` — остановить и отключиться\n"
            "`!скип` — следующий трек\n"
            "`!найди <название>` — поиск\n"
            "`!включи <номер>` — включить результат поиска\n"
            "`!очередь` — текущий трек и история\n"
            "`!обновить` — пересканировать коллекцию\n\n"
            "**🎙️ Голос:**\n"
            "`Вонючка, стоп`\n"
            "`Вонючка, играй`\n"
            "`Вонючка, скип`\n"
            "`Вонючка, включи черную гору`"
        )


# ============================================================
# GLOBAL RADIO INSTANCE
# ============================================================

radio = RadioManager()


# ============================================================
# CACHE REFRESH
# ============================================================


@tasks.loop(minutes=CACHE_REFRESH_MINUTES)
async def refresh_cache_loop() -> None:
    await radio.scan_collection()


@refresh_cache_loop.before_loop
async def before_refresh_cache_loop() -> None:
    await bot.wait_until_ready()


# ============================================================
# BOT EVENTS
# ============================================================


@bot.event
async def on_ready() -> None:
    radio._loop = asyncio.get_running_loop()

    log.info("[BOT] 🤖 Бот вошёл как %s", bot.user)
    log.info("[BOT] 🆔 Bot ID: %s", bot.user.id)

    # Load Whisper off the event loop.
    await asyncio.to_thread(load_whisper_model)

    # Initial collection scan.
    await radio.scan_collection()

    if not refresh_cache_loop.is_running():
        refresh_cache_loop.start()


# ============================================================
# COMMANDS
# ============================================================


def in_command_channel(ctx: commands.Context) -> bool:
    return ctx.channel.id == COMMAND_CHANNEL_ID


@bot.command(name="слушай")
async def listen_command(ctx: commands.Context) -> None:
    if not in_command_channel(ctx):
        return
    await radio.command_listen(ctx)


@bot.command(name="стоп")
async def stop_command(ctx: commands.Context) -> None:
    if not in_command_channel(ctx):
        return
    await radio.command_stop(ctx)


@bot.command(name="скип")
async def skip_command(ctx: commands.Context) -> None:
    if not in_command_channel(ctx):
        return
    await radio.command_skip(ctx)


@bot.command(name="найди")
async def find_command(
    ctx: commands.Context,
    *,
    query: str | None = None,
) -> None:
    if not in_command_channel(ctx):
        return

    if not query:
        await ctx.send(
            "❌ Напиши название: `!найди черная гора`"
        )
        return

    await radio.command_find(ctx, query)


@bot.command(name="включи")
async def play_command(
    ctx: commands.Context,
    number: int | None = None,
) -> None:
    if not in_command_channel(ctx):
        return

    if number is None:
        await ctx.send(
            "❌ Напиши номер: `!включи 1`"
        )
        return

    await radio.command_play(ctx, number)


@bot.command(name="очередь")
async def queue_command(ctx: commands.Context) -> None:
    if not in_command_channel(ctx):
        return
    await radio.command_queue(ctx)


@bot.command(name="обновить")
async def update_command(ctx: commands.Context) -> None:
    if not in_command_channel(ctx):
        return
    await radio.command_update(ctx)


@bot.command(name="помощь")
async def help_command(ctx: commands.Context) -> None:
    if not in_command_channel(ctx):
        return
    await radio.command_help(ctx)


# ============================================================
# COMMAND ERROR HANDLER
# ============================================================


@bot.event
async def on_command_error(
    ctx: commands.Context,
    error: commands.CommandError,
) -> None:
    if ctx.channel.id != COMMAND_CHANNEL_ID:
        return

    if isinstance(error, commands.CommandNotFound):
        return

    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ Не хватает аргумента.")
        return

    if isinstance(error, commands.BadArgument):
        await ctx.send("❌ Неверный аргумент.")
        return

    log.exception("[BOT] ❌ Ошибка команды", exc_info=error)


# ============================================================
# START
# ============================================================


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError(
            "Не задана переменная окружения DISCORD_TOKEN. "
            "В PowerShell: $env:DISCORD_TOKEN = \"YOUR_NEW_TOKEN\""
        )

    if not os.path.isfile(FFMPEG_PATH):
        raise FileNotFoundError(
            f"FFmpeg не найден: {FFMPEG_PATH}"
        )

    log.info("[BOT] 🚀 Запускаю Вонючку")
    bot.run(DISCORD_TOKEN)
