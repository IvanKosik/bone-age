import bsmu.vision.app.main as parent_app


def run_app(childs=()):
    print('Run, Bone Age! Run!')

    parent_app.run_app(childs + (__file__,))
    # bsmu.vision.app.main.run_app((*childs, __file__))


if __name__ == '__main__':
    run_app()
