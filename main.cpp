#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include<ceres/ceres.h>


//构建代价函数结构体，abc为待优化参数，residual为残差。
double calc_d(double a, double b,double c,double x0,double y0,double z0, double x1, double y1, double z1){
	return (a*x0-a*x1+b*y0-b*y1+c*z0-c*z1)/(a*a+b*b+c*c);
}
//点线距离
double dist(double x0, double x1, double y0,double y1,double z0, double z1, double a,double b, double c){
	double d;
	d = calc_d(a,b,c,x0,y0,z0,x1,y1,z1);
	return sqrt((x0-a*d-x1)*(x0-a*d-x1) + (y0-b*d-y1)*(y0-b*d-y1) + (z0-c*d-z1)*(z0-c*d-z1));
}

struct CURVE_FITTING_COST
{
  CURVE_FITTING_COST(double x,double y,double z):x0(x),y0(y),z0(z){}
  template <typename T>
  bool operator()(const T* const abc,T* residual)const
  {
  	T a,b,c,x1,y1,z1,d;
  	x1 = abc[0];
  	y1 = abc[1];
  	z1 = abc[2];
  	a = abc[3];
  	b = abc[4];
  	c = abc[5];
  	d = (a*x0-a*x1+b*y0-b*y1+c*z0-c*z1)/(a*a+b*b+c*c);
  	residual[0] = ceres::sqrt((x0-a*d-x1)*(x0-a*d-x1) + (y0-b*d-y1)*(y0-b*d-y1) + (z0-c*d-z1)*(z0-c*d-z1));
    return true;
  }
  const double x0,y0,z0;
};



int main() {
	Eigen::Vector3d px,pz,py;
	Eigen::Matrix3d rotate;
	//1.从点云中确定 x 轴的方向
	//参数初始化设置，abc初始化为0
	double abc[6]={11,0,0,20,0,3}; //x y z a b c
	pcl::PointCloud<pcl::PointXYZ> linePoint;
	pcl::PointCloud<pcl::PointXYZ> lineout;
	pcl::io::loadPCDFile<pcl::PointXYZ> ("x.pcd", linePoint);
	ceres::Problem problem;
	for (int j = 0; j < linePoint.size(); ++j) {
		std::cout<<linePoint[j].x<<std::endl;
		std::cout<<linePoint[j].y<<std::endl;
		std::cout<<linePoint[j].z<<std::endl<<std::endl;
		//残差的维度为 1（距离） 优化的维度为 6
		problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,6>(
        new CURVE_FITTING_COST(linePoint[j].x,linePoint[j].y,linePoint[j].z)),nullptr,abc);
	}
	ceres::Solver::Options options;
  	options.linear_solver_type=ceres::DENSE_QR;
  	options.minimizer_progress_to_stdout=true;
  	ceres::Solver::Summary summary;
  	ceres::Solve(options,&problem,&summary);
	std::cout<<"x= "<<abc[0]<<std::endl;
	std::cout<<"y= "<<abc[1]<<std::endl;
	std::cout<<"z= "<<abc[2]<<std::endl;
	std::cout<<"a= "<<abc[3]<<std::endl;
	std::cout<<"b= "<<abc[4]<<std::endl;
	std::cout<<"c= "<<abc[5]<<std::endl;
	px = Eigen::Vector3d(50.546,0,6.2455);
	pz = Eigen::Vector3d(-0.131814,0.0288355,0.990855);
	py = pz.cross(px);
	double d;
	d = abc[1]/abc[4];
	for (double k = 0; k < 100; ++k) {
		pcl::PointXYZ temp;
		temp.x = k/10;
		temp.y = abc[4]*d + abc[1];
		temp.z = abc[5]*d + abc[2];
		lineout.push_back(temp);
	}
	pcl::io::savePCDFile("lineout.pcd",lineout);
    return 0;
}