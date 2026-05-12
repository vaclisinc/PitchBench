import argspace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",      action="store_true")
    parser.add_argument("--sources",      nargs="+", metavar="SRC", default=None,
                        help=f"Sources to run (default: all). Available: {ALL_SOURCES}")
    parser.add_argument("--models",       nargs="+", metavar="MODEL",
                        help="Model URLs (http://...) or slugs (openrouter/..., dashscope/...)")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args
