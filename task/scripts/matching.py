#!/usr/bin/env python
# This Python file uses the following encoding: utf-8
import rospy
from task.msg import Res
from task.msg import Det

def callback(data):
    mastertext=u'Родина'.encode('utf-8')
    print(mastertext)
    rospy.loginfo(data.text)
    pub=rospy.Publisher('/final_result', Res, queue_size=10)
    rate = rospy.Rate(10)
    msg=Res()
    msg.centre=data.centre
    if data.text==mastertext:
        msg.res=True
    else:
        msg.res=False
    print(msg.res)
    pub.publish(msg)
    rate.sleep()
    

if __name__ == '__main__':
    rospy.init_node('matching', anonymous=True)
    sub=rospy.Subscriber('/detection_result',Det,callback)
    rospy.spin()