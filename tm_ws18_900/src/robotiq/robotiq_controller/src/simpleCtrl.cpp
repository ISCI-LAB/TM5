#include "ros/ros.h"
#include <sstream>
#include "robotiq_controller/Robotiq2FGripper_robot_output.h"

robotiq_controller::Robotiq2FGripper_robot_output command;

using namespace std;

void reset_gripper()
{
	command.rACT = 0;
    command.rGTO = 0;
    command.rSP  = 0;
    command.rFR  = 0;	
}
void init_gripper()
{
	command.rACT = 1;
    command.rGTO = 1;
    command.rSP  = 150;
    command.rFR  = 0;
}

void set_gripper(int mode)
{
	if(mode == 0)	//Open
	{
		command.rACT = 1;
    	command.rGTO = 1;
    	command.rSP  = 150;
    	command.rFR  = 0;
		command.rPR = 0;
	}
	if(mode == 1)	//Close
	{
		command.rACT = 1;
    	command.rGTO = 1;
    	command.rSP  = 150;
    	command.rFR  = 0;
		command.rPR = 255;
	}
}

int main(int argc,char **argv)
{
	int state = 0;

	ros::init(argc,argv,"simpleCtrl");
	ros::NodeHandle n;
	
	ros::Publisher pub_ctrl = n.advertise<robotiq_controller::Robotiq2FGripper_robot_output>("Robotiq2FGripperRobotOutput",1000);
	ros::Rate loop_rate(10);

	
	while(ros::ok())
	{
		if(state == 0)
		{
			cout << "Activating gripper,press enter to continue" <<endl;
			//getchar();
			sleep(1);
			reset_gripper();
			state = 1;
			pub_ctrl.publish(command);
			getchar();
		}
		else if(state == 1)
		{
			cout << "/---Gripper close---/" <<endl;
			set_gripper(1);
			state = 2;
			pub_ctrl.publish(command);
			getchar();
		}
		else if(state == 2)
		{
			cout << "/---Gripper open---/" <<endl;
			set_gripper(0);	
			state = 1;
			pub_ctrl.publish(command);
			getchar();
		}
		//pub_ctrl.publish(command);
		ros::spinOnce(); 
		loop_rate.sleep();
	}
	return 0;
}
