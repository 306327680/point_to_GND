#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include<ceres/ceres.h>
#include <pcl/registration/ndt.h>      				//NDT(正态分布)配准类头文件
#include <pcl/filters/approximate_voxel_grid.h>     //滤波类头文件  （使用体素网格过滤器处理的效果比较好）
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/extract_indices.h>
Eigen::Vector3d px,pz(-0.131814,0.0288355,0.990855),py; //平面法向量

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
//拟合一个和平面法向量垂直的 x轴
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
  	d = (a*x0 - a*x1 + b*y0 - b*y1 + c*z0 - c*z1) / (a*a + b*b + c*c);
  	residual[0] = ceres::sqrt((x0-a*d-x1)*(x0-a*d-x1) + (y0-b*d-y1)*(y0-b*d-y1) + (z0-c*d-z1)*(z0-c*d-z1));
  	residual[1] = abc[3]*pz[0] +abc[4]*pz[1]+abc[5]*pz[2]; //保证和平面法向量正交的约束
    return true;
  }
  const double x0,y0,z0;
};

//拟合两个与两个轴垂直的法向量
struct VERTICLE_FITTING
{
	VERTICLE_FITTING(double x,double y,double z,double x1,double y1,double z1):x0(x),y0(y),z0(z),x1(x1),y1(y1),z1(z1){}
	template <typename T>
	bool operator()(const T* const abc,T* residual)const
	{
		
		residual[0] = x0*abc[0]+y0*abc[1]+z0*abc[2];
		residual[1] = x1*abc[0]+y1*abc[1]+z1*abc[2];
		return true;
	}
	const double x0,y0,z0,x1,y1,z1;
};

//输入构造:两点 输入:RT 残差:两点距离
struct AxisAxisFitting
{
	AxisAxisFitting(Eigen::Vector3d origin_pt, Eigen::Vector3d target_pt) : origin_pt(origin_pt), target_pt(target_pt) {
	}
	
	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};// rotation
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};    // translation
		Eigen::Matrix<T, 3, 1> cp{T(origin_pt.x()), T(origin_pt.y()), T(origin_pt.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;//旋转平移到
//		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = ceres::sqrt((point_w[0]-target_pt.x())*(point_w[0]-target_pt.x()) + (point_w[1]-target_pt.y())*(point_w[1]-target_pt.y()) + (point_w[2]-target_pt.z())* (point_w[2]-target_pt.z()));
		return true;
	}
	
	static ceres::CostFunction *Create(const Eigen::Vector3d origin_pt, const Eigen::Vector3d target_pt){
		return (new ceres::AutoDiffCostFunction<AxisAxisFitting, 1, 4, 3>(new AxisAxisFitting(origin_pt, target_pt)));
	}
	
	Eigen::Vector3d origin_pt;
	Eigen::Vector3d target_pt;
};
// 分割点云的地面部分
pcl::PointCloud<pcl::PointXYZI> sacPlaneExtract(pcl::PointCloud<pcl::PointXYZI> cloud_filtered1,Eigen::Vector3d& normal_vector){
	pcl::SACSegmentation<pcl::PointXYZI> seg;
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
 	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	*cloud_filtered = cloud_filtered1;
	//1.分割
	// Optional
	seg.setOptimizeCoefficients (true);
	// Mandatory
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setDistanceThreshold (0.1);
	seg.setInputCloud (cloud_filtered);
	seg.segment (*inliers, *coefficients);
	std::cout << "Model coefficients: " << coefficients->values[0] << " "
			  << coefficients->values[1] << " "
			  << coefficients->values[2] << " "
			  << coefficients->values[3] <<std::endl;
	normal_vector = Eigen::Vector3d(coefficients->values[0],coefficients->values[1],coefficients->values[2]);
	//2.过滤
	for (int j = 0; j < inliers->indices.size(); ++j) {
		cloud_projected->push_back(cloud_filtered->points[inliers->indices[j]]);
	}
	return *cloud_projected;
}
pcl::PointCloud<pcl::PointXYZI> getxPCD(pcl::PointCloud<pcl::PointXYZI> sacpoints){
	pcl::PointCloud<pcl::PointXYZI> result;
	for (int i = 0; i < sacpoints.size(); ++i) {
		if (fabs(sacpoints.points[i].y)<0.01)
			result.push_back(sacpoints.points[i]);
	}
	return result;
}
int main() {
 	//0.分割点云
	pcl::PointCloud<pcl::PointXYZI> pointRaw;
	pcl::PointCloud<pcl::PointXYZI> sacpoints;
	pcl::io::loadPCDFile<pcl::PointXYZI> ("origin.pcd", pointRaw);
	sacpoints = sacPlaneExtract(pointRaw,pz);
	pcl::io::savePCDFile("ransas.pcd",sacpoints);
	pcl::io::savePCDFile("x.pcd",getxPCD(sacpoints));
	Eigen::Matrix3d rotate;
	//1.从点云中确定 x 轴的方向
	//参数初始化设置，abc初始化为0
	double abc[6]={11,0,0,20,0,3}; //x y z a b c 点向式
	pcl::PointCloud<pcl::PointXYZ> linePoint;
	pcl::PointCloud<pcl::PointXYZ> lineout;
	pcl::io::loadPCDFile<pcl::PointXYZ> ("x.pcd", linePoint);
	abc[0] = linePoint[0].x;
	abc[1] = linePoint[1].x;
	abc[2] = linePoint[2].x;
	ceres::Problem problem;
	for (int j = 0; j < linePoint.size(); ++j) {
		//残差的维度为 1（距离） 优化的维度为 6
		//1.resident 2. methord 3. which to opti
		//1 new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,2,6>(new CURVE_FITTING_COST(linePoint[j].x,linePoint[j].y,linePoint[j].z))   ->AutoDiffCostFunction
		//1.1 new CURVE_FITTING_COST(linePoint[j].x,linePoint[j].y,linePoint[j].z) 															-> CURVE_FITTING_COST
		problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,2,6>(new CURVE_FITTING_COST(linePoint[j].x,linePoint[j].y,linePoint[j].z)) ,
		        (new ceres::CauchyLoss(0.5)),   abc);//使用不同的loss function
	}
	ceres::Solver::Options options;
  	options.linear_solver_type=ceres::DENSE_QR;
  	options.minimizer_progress_to_stdout= false;
  	ceres::Solver::Summary summary;
  	ceres::Solve(options,&problem,&summary);
	std::cout<<" x= "<<abc[0]<<std::endl;
	std::cout<<" y= "<<abc[1];
	std::cout<<" z= "<<abc[2];
	std::cout<<" a= "<<abc[3];
	std::cout<<" b= "<<abc[4];
	std::cout<<" c= "<<abc[5]<<std::endl;;
	px = Eigen::Vector3d(50.546,0,6.2455);
	py = pz.cross(px);
	double d;
	
	for (double k = -100; k < 200; ++k) {
		pcl::PointXYZ temp;
		temp.x = k/10;
		d = (k/10-abc[0])/abc[3];
		temp.y = abc[4]*d + abc[1];
		temp.z = abc[5]*d + abc[2];
		lineout.push_back(temp);
	}
	
	pcl::io::savePCDFile("lineout.pcd",lineout);
	Eigen::Vector3d x_aixs,z_axis;
	x_aixs = Eigen::Vector3d(abc[3],abc[4],abc[5]);
	std::cout<<"dot: "<<x_aixs.dot(pz)<<std::endl;// 得到这个直线
	//2. 计算法向量的直线方向
	ceres::Problem problem1;
	double y_axis[3] = {1,1,1};
	problem1.AddParameterBlock(y_axis,3);
	ceres::CostFunction *cost_function =  new ceres::AutoDiffCostFunction<VERTICLE_FITTING,2,3>(new VERTICLE_FITTING(abc[3],abc[4],abc[5],pz[0],pz[1],pz[2]));
	ceres::LossFunction *loss_function1 = new ceres::CauchyLoss(0.5); //2.1 设定 loss function
	problem1.AddResidualBlock(cost_function,loss_function1,y_axis);
	problem1.AddParameterBlock(y_axis,3);
	
	ceres::Solve(options,&problem1,&summary);
	std::cout<<" y轴法向量 x= "<<y_axis[0]<<" y= "<<y_axis[1]<<" z= "<<y_axis[2]<<std::endl;
	
	//3. 解算真正的tf
	// 计算4 个坐标点之间的距离 (1).坐标原点 (2). 到(1,0,0) (3). 到 (0,1,0) (4) 到 (0,0,1)
	
	double parameters[7] = {1, 2, 3, 4, 5, 6, 7};
	// Map类用于通过C++中普通的连续指针或者数组 （raw C/C++ arrays）来构造Eigen里的Matrix类，这就好比Eigen里的Matrix类的数据和raw C++array 共享了一片地址，也就是引用。
//	1. 比如有个API只接受普通的C++数组，但又要对普通数组进行线性代数操作，那么用它构造为Map类，直接操作Map就等于操作了原始普通数组，省时省力。
//	2. 再比如有个庞大的Matrix类，在一个大循环中要不断读取Matrix中的一段连续数据，如果你每次都用block operation 去引用数据，太累（虽然block operation 也是引用类型）。
//	于是就事先将这些数据构造成若干Map，那么以后循环中就直接操作Map就行了。实际上Map类并没有自己申请一片空内存，只是一个引用，所以需要构造时初始化，或者使用Map的指针。引申一下，
//	Eigen里 ref 类也是引用类型，Armadillo 里 subview 都是引用类型，Eigen开发人说的The use 'sub' as a Matrix or Map. Actually Map,
//	Ref, and Block inherit from the same base class. You can also use Block.所以说了这么多，就一句话 Map 就是个引用。
	ceres::Problem problem2;
	ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
	ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1); //2.1 设定 loss function
	problem2.AddParameterBlock(parameters, 4, q_parameterization);//rotation
	problem2.AddParameterBlock(parameters + 4, 3);//translation
	Eigen::Vector3d p1,p2,p3,p4,a1,a2,a3,a4;
	p1 = Eigen::Vector3d (0,0,0);
	p2 = Eigen::Vector3d (1,0,0);
	p3 = Eigen::Vector3d (0,1,0);
	p4 = Eigen::Vector3d (0,0,1);
	std::cout<<"原点坐标: "<<0<<" "<<abc[4]*(-abc[0]/abc[3]) + abc[1]<<" "<<abc[5]*(-abc[0]/abc[3]) + abc[1]<<std::endl;
	a1 = Eigen::Vector3d (0,abc[4]*(-abc[0]/abc[3]) + abc[1],abc[5]*(-abc[0]/abc[3]) + abc[1]);
	//pz
	pz.normalize();
	a4 = a1 + pz;//z轴
	Eigen::Vector3d temp_a3(y_axis[0],y_axis[1],y_axis[2]),temp_a4(abc[0],abc[1],abc[2]);//y轴
	temp_a3.normalize();
	a3 = a1 + temp_a3;
	Eigen::Vector3d x_axis(abc[3],abc[4],abc[4]);
	x_aixs.normalize();
	a2 = a1 + x_aixs;
	std::cout<<"a1: "<<a1<<"\n a2: \n"<<a2<<"\n a3: \n"<<a3<<"\n a4: \n"<<a4<<std::endl;
	ceres::CostFunction *cost_function_1 =  AxisAxisFitting::Create(p1,a1);
	ceres::CostFunction *cost_function_2 =  AxisAxisFitting::Create(p2,a2);
	ceres::CostFunction *cost_function_3 =  AxisAxisFitting::Create(p3,a3);
	ceres::CostFunction *cost_function_4 =  AxisAxisFitting::Create(p4,a4);
	
	problem2.AddResidualBlock(cost_function_1,loss_function,parameters,parameters+4);
	problem2.AddResidualBlock(cost_function_2,loss_function,parameters,parameters+4);
	problem2.AddResidualBlock(cost_function_3,loss_function,parameters,parameters+4);
	problem2.AddResidualBlock(cost_function_4,loss_function,parameters,parameters+4);
	ceres::Solve(options,&problem2,&summary);
	Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);//放的是指针就比较方便了, 里面的随便改也不影响值
	Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

	std::cout << t_w_curr << std::endl;
	std::cout <<"x: "<< q_w_curr.x()<<" y: "<<q_w_curr.y()<<" z: "<<q_w_curr.z()<<" w: "<<q_w_curr.w() << std::endl;
	pcl::PointCloud<pcl::PointXYZ> tfed;
	pcl::io::loadPCDFile<pcl::PointXYZ> ("origin.pcd", tfed);
	Eigen::Isometry3d test;
	test.setIdentity();
	test.translate(t_w_curr);
	test.rotate(q_w_curr);
	pcl::transformPointCloud(tfed,tfed,test.inverse().matrix());
	std::cout<<"校正后的矩阵为: \n"<<test.inverse().matrix()<<std::endl;
	pcl::io::savePCDFile("tfed.pcd",tfed);
    return 0;
}