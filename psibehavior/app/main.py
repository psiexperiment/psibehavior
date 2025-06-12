from enaml.qt.qt_application import QtApplication
import enaml
with enaml.imports():
    from .app import Main


# Make sure this is imported to register the paradigms
from psibehavior import paradigms


def abts():
    import argparse
    parser = argparse.ArgumentParser('abts')
    parser.add_argument('config', nargs='?')

    args = parser.parse_args()
    app = QtApplication()
    view = Main()

    # This needs to be loaded to ensure that some defaults are set properly.
    view.settings.load_config(args.config)

    view.show()
    app.start()
    return True


if __name__ == '__main__':
    abts()
