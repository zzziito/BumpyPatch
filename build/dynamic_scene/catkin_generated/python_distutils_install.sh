#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/rtlink/jiwon/bumpypatch_ws/src/dynamic_scene"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/rtlink/jiwon/bumpypatch_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/rtlink/jiwon/bumpypatch_ws/install/lib/python3/dist-packages:/home/rtlink/jiwon/bumpypatch_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/rtlink/jiwon/bumpypatch_ws/build" \
    "/usr/bin/python3" \
    "/home/rtlink/jiwon/bumpypatch_ws/src/dynamic_scene/setup.py" \
     \
    build --build-base "/home/rtlink/jiwon/bumpypatch_ws/build/dynamic_scene" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/rtlink/jiwon/bumpypatch_ws/install" --install-scripts="/home/rtlink/jiwon/bumpypatch_ws/install/bin"
