import subprocess

def make_animation(identifier, moviename):
    subprocess.run(
        [ "ffmpeg",
          "-y",
          "-framerate", "10",
          "-start_number", "0",
          "-i", identifier+"%d.png",
          "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
          "-c:v", "libx264",
          "-pix_fmt", "yuv420p",
          moviename,
        ], check=True
    )