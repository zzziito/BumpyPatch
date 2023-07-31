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

echo_and_run cd "/home/rtlink/jiwon/paper_ws/src/bumpypatch"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/rtlink/jiwon/paper_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/rtlink/jiwon/paper_ws/install/lib/python3/dist-packages:/home/rtlink/jiwon/paper_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/rtlink/jiwon/paper_ws/build" \
    "/home/rtlink/anaconda3/bin/python3" \
    "/home/rtlink/jiwon/paper_ws/src/bumpypatch/setup.py" \
     \
    build --build-base "/home/rtlink/jiwon/paper_ws/build/bumpypatch" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/rtlink/jiwon/paper_ws/install" --install-scripts="/home/rtlink/jiwon/paper_ws/install/bin"
