import sys
import ast
import argparse
import logging
import numpy as np
from pathlib import Path
from webvtt import WebVTT
from dataclasses import dataclass
from stable_whisper import load_faster_whisper, WhisperResult
from stable_whisper.audio import load_audio
from typing import Any, Union
from tqdm.auto import tqdm

logger = logging.getLogger(__file__)

@dataclass
class AlignConfig:
    audio: Union[Path, np.ndarray]
    vtt: WebVTT
    text: str
    model: Any
    language: str
    verbose: bool = False
    offset: float = 0.0

def align(cfg: AlignConfig) -> WhisperResult:
    """Simpler case is to let stable-whisper do the alignment all by itself."""
    audio = cfg.audio.as_posix() if isinstance(cfg.audio, Path) else cfg.audio
    result = cfg.model.align(
        audio,
        cfg.text,
        language=cfg.language,
        fast_mode=False,
        verbose=cfg.verbose,
        regroup=True,
        suppress_silence=True,
    )
    if cfg.offset != 0:
        result.offset_time(cfg.offset)
    return result

def split_per_line(cfg: AlignConfig):
    """More complicated case needs to read audio, split by timestamps, align, then combine."""
    sampling_rate = 16_000
    audio = load_audio(cfg.audio.as_posix(), sr=sampling_rate)
    for cap in cfg.vtt.captions:
        start = int(cap.start_in_seconds * sampling_rate)
        end = int(cap.end_in_seconds * sampling_rate)
        yield AlignConfig(
            audio=audio[start:end],
            text=cap.text,
            vtt=None,
            model=cfg.model,
            language=cfg.language,
            verbose=None,
            offset=cap.start_in_seconds
        )


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to test data folder containing paired audio and subtitles.")
    parser.add_argument("--model", default="base", help="Model str for faster-whisper, can be size e.g. 'base' or local path. Default: 'base'")
    parser.add_argument("--model_args", default="{}", help="Python dict as a string containing extra args to pass to faster whisper.")
    parser.add_argument("--audio_formats", default="mp4,mp3,wav", help="Comma separated list of audio formats. Default: mp4,mp3,wav")
    parser.add_argument("--language", default="zh", help="Supplying the language code should improve alignment performance.")
    parser.add_argument("--per_line", action="store_true", help=(
        "Align the audio per line. CAUTION: It's almost always worse to enable this."
        " Only when 1.You get timestamp shifts with default mode"
        " 2.You are SURE each line's timestamp in the subtitle file fully contains the line."
        " 3.You are SURE each line in the subtitle is long enough (more than a few seconds)."
        " should you enable this. Definitely "
        " will get 'Failed to align' if either 2 or 3 is False."
    ))
    args = parser.parse_args()
    folder = Path(args.folder)
    model = load_faster_whisper(
        args.model,
        **ast.literal_eval(args.model_args)
    )
    data_pairs = []
    for fmt in args.audio_formats.split(","):
        fmt = fmt.strip().strip(".").strip("*")
        for audio in folder.glob(f"*.{fmt}"):
            subs = audio.with_suffix(".srt")
            if subs.exists():
                vtt = WebVTT.from_srt(subs)
            else:
                subs = audio.with_suffix(".vtt")                
                if subs.exists():
                    vtt = WebVTT.read(subs)
                else:
                    raise FileNotFoundError(f"Neither .vtt nor .srt are found for file {audio.name}.")
            data_pairs.append(AlignConfig(
                audio=audio,
                text=" ".join(c.text for c in vtt.captions),
                vtt=vtt,
                model=model,
                language=args.language,
            ))

    logger.info("Detected %d audios to align.", len(data_pairs))

    for cfg in data_pairs:
        logger.info("Processing: %s", cfg.audio)
        if args.per_line:
            segments = []
            for sub_cfg in tqdm(split_per_line(cfg), total=len(cfg.vtt.captions)):
                r = align(sub_cfg)
                segments.extend(r.segments_to_dicts())
            result = WhisperResult(segments)
        else:
            result = align(cfg)
        result.save_as_json(cfg.audio.with_suffix(".json").as_posix())

    logger.info("Success.")

if __name__ == "__main__":
    main()