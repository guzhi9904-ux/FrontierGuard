from frontierguard.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["solve-map", *__import__("sys").argv[1:]]))
