#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include<ceres/ceres.h>


//构建代价函数结构体，abc为待优化参数，residual为残差。
struct CURVE_FITTING_COST
{
  CURVE_FITTING_COST(double x,double y):_x(x),_y(y){}
  template <typename T>
  bool operator()(const T* const abc,T* residual)const
  {
    residual[0]=_y-ceres::exp(abc[0]*_x*_x+abc[1]*_x+abc[2]);
    return true;
  }
  const double _x,_y;
};



int main() {
	Eigen::Vector3d px,pz,py;
	Eigen::Matrix3d rotate;
	//1.从点云中确定 x 轴的方向
	//参数初始化设置，abc初始化为0
	double a=3,b=2,c=1;
	double abc[3]={0,0,0};
	
	px = Eigen::Vector3d(50.546,0,6.2455);
	pz = Eigen::Vector3d(-0.131814,0.0288355,0.990855);
	py = pz.cross(px);
    std::cout << px.dot(pz) << std::endl;
	
    return 0;
}