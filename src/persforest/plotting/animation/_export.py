"""Animation writers and assembly of PNG frames with ffmpeg."""

from pathlib import Path
from typing import TYPE_CHECKING
import shutil
import subprocess

if TYPE_CHECKING:
    from matplotlib.animation import Animation
    from matplotlib.figure import Figure

def _ffmpeg_path_or_raise() -> str:
    """
    Return ffmpeg executable path, or raise with a clear install hint.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install ffmpeg to export MP4 "
            "(for example: `brew install ffmpeg`)."
        )
    return ffmpeg_path

def _assemble_mp4_with_ffmpeg(
    *,
    frame_dir: Path,
    fps: int,
    output_file: Path,
) -> None:
    """
    Assemble PNG frames into an MP4 via ffmpeg.
    """
    ffmpeg = _ffmpeg_path_or_raise()
    input_pattern = str(frame_dir / "frame_%04d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-r",
        str(int(fps)),
        "-i",
        input_pattern,
        "-vcodec",
        "libx264",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(output_file),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed while assembling MP4:\n{err}") from e

def _save_filtration_animation(
    anim: "Animation", fig: "Figure", filename: str | Path, *, fps: int, dpi: int,
) -> None:
    """Save an animation with the existing _animate_filtration_generic writer settings.

    Parameters
    ----------
    anim : matplotlib.animation.Animation
        Animation to save.
    fig : matplotlib.figure.Figure
        Figure to draw before saving and use for background color.
    filename : str or pathlib.Path
        Output path; its extension selects the writer.
    fps : int
        Frames per second.
    dpi : int
        Output resolution.
    """
    from matplotlib.animation import FFMpegWriter

    fig.canvas.draw()


    fname = str(filename)
    ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""

    savefig_kwargs = {
        "pad_inches": 0.0,
        "facecolor": fig.get_facecolor(),
    }

    if ext == "mp4":
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(fname, writer=writer, dpi=dpi, savefig_kwargs=savefig_kwargs)
    elif ext in {"gif", "gifv"}:
        try:
            from matplotlib.animation import PillowWriter
        except ImportError as e:
            raise RuntimeError(
                "Saving as GIF requires Pillow. Install it with `pip install pillow` "
                "or `pip install \".[animation]\"`."
            ) from e
        writer = PillowWriter(fps=fps)
        anim.save(fname, writer=writer, dpi=dpi, savefig_kwargs=savefig_kwargs)
    else:
        anim.save(fname, dpi=dpi, fps=fps, savefig_kwargs=savefig_kwargs)


def _save_paired_animation(
    anim: "Animation", filename: str | Path, *, fps: int, dpi: int,
) -> None:
    """Save an animation with the existing animate_filtration_pair writer settings.

    Parameters
    ----------
    anim : matplotlib.animation.Animation
        Animation to save.
    filename : str or pathlib.Path
        Output path; its extension selects the writer.
    fps : int
        Frames per second.
    dpi : int
        Output resolution.
    """
    from matplotlib.animation import FFMpegWriter

    fname = str(filename)
    ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
    if ext == "mp4":
        writer = FFMpegWriter(
            fps=fps,
            #bitrate=2000,
            codec="libx264",
            extra_args=[
                "-pix_fmt", "yuv420p",
                "-vf", "scale=1920:-2",
                "-profile:v", "baseline",
                "-level", "3.1",
                "-movflags", "+faststart",
            ],)
        anim.save(fname, writer=writer, dpi=dpi)
    elif ext in {"gif", "gifv"}:
        try:
            from matplotlib.animation import PillowWriter
        except ImportError as e:
            raise RuntimeError(
                "Saving as GIF requires Pillow. Install it with `pip install pillow` "
                "or `pip install \".[animation]\"`."
            ) from e
        writer = PillowWriter(fps=fps)
        anim.save(fname, writer=writer, dpi=dpi)
    else:
        # Fallback to default writer chosen by matplotlib
        anim.save(fname, dpi=dpi, fps=fps)


def _save_measurement_animation(
    anim: "Animation", filename: str | Path, *, fps: int, dpi: int,
) -> None:
    """Save an animation with the existing animate_barcode_measurement_generic writer settings.

    Parameters
    ----------
    anim : matplotlib.animation.Animation
        Animation to save.
    filename : str or pathlib.Path
        Output path; its extension selects the writer.
    fps : int
        Frames per second.
    dpi : int
        Output resolution.
    """
    from matplotlib.animation import FFMpegWriter

    fname = str(filename)
    ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
    if ext == "mp4":
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(fname, writer=writer, dpi=dpi)
    elif ext in {"gif", "gifv"}:
        try:
            from matplotlib.animation import PillowWriter
        except ImportError as e:  # optional dependency
            raise RuntimeError(
                "Saving as GIF requires Pillow. Install it with `pip install pillow` "
                "or `pip install \".[animation]\"`."
            ) from e
        writer = PillowWriter(fps=fps)
        anim.save(fname, writer=writer, dpi=dpi)
    else:
        # Fallback to default writer chosen by matplotlib
        anim.save(fname, dpi=dpi, fps=fps)

