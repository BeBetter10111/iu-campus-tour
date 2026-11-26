import argparse
import logging
import sys

import config


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="IU Campus Virtual Tour")
    p.add_argument("--strict", action="store_true",
                   help="loi neu thieu du lieu OSM thay vi dung canh thu nghiem")
    p.add_argument("--debug", action="store_true", help="bat GL error checking va log chi tiet")
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--msaa", type=int, choices=(0, 2, 4, 8), help="so mau MSAA")
    p.add_argument("--fps-cap", type=int, help="gioi han FPS (0 = khong gioi han)")
    p.add_argument("--size", metavar="WxH", help="kich thuoc cua so, vd 1920x1080")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.no_vsync:
        config.VSYNC = False
    if args.msaa is not None:
        config.MSAA_SAMPLES = args.msaa
    if args.fps_cap is not None:
        config.FPS_CAP = args.fps_cap
    if args.debug:
        config.GL_ERROR_CHECKING = True
    if args.size:
        config.WINDOW_WIDTH, config.WINDOW_HEIGHT = (int(v) for v in args.size.lower().split("x"))

    import OpenGL
    OpenGL.ERROR_CHECKING = config.GL_ERROR_CHECKING

    try:
        from core.app import App
        App(strict=args.strict).run()
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Ung dung gap loi nghiem trong")
        print("\nLoi: khong the chay ung dung. Xem chi tiet o tren (chay them --debug).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
