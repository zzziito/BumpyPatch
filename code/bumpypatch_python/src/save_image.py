import rospy
from bumpypatch.srv import SaveImage

rospy.init_node('image_saver_client')

rospy.wait_for_service('save_image')
try:
    save_image = rospy.ServiceProxy('save_image', SaveImage)
    resp = save_image()
    print("Image saved:", resp.success)
except rospy.ServiceException as e:
    print("Service call failed:", e)
