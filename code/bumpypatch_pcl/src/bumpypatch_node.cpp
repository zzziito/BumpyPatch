#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <cmath>
#include <vector>
#include <pcl_ros/transforms.h>

#define MIN_POINTS 10

double min_range_z2_ = 12.3625;
double min_range_z3_ = 22.025;
double min_range_z4_ = 41.35;
double max_range_ = 100.0;

std::vector<double> ring_sizes_ = {min_range_z2_, min_range_z3_, min_range_z4_, (max_range_ - min_range_z4_)};
std::vector<double> sector_sizes_ = {2 * M_PI / 16, 2 * M_PI / 16, 2 * M_PI / 16, 2 * M_PI / 16};

ros::Publisher pub;

struct Zone {
    pcl::PointCloud<pcl::PointXYZI> points;
};

inline double xy2radius(double x, double y) {
    return sqrt(x*x + y*y);
}

inline double xy2theta(double x, double y) {
    return atan2(y, x);
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    std::vector<Zone> czm(4*16); // for 4 rings and 16 sectors

    for (auto const &pt : cloud->points) {
        int ring_idx, sector_idx;
        double r = xy2radius(pt.x, pt.y);
        if ((r <= max_range_) && (r > min_range_z2_)) {
            double theta = xy2theta(pt.x, pt.y);

            if (r < min_range_z2_) {
                ring_idx   = static_cast<int>(((r - min_range_z2_) / ring_sizes_[0]));
                sector_idx = static_cast<int>((theta / sector_sizes_[0]));
            } else if (r < min_range_z3_) {
                ring_idx   = static_cast<int>(((r - min_range_z2_) / ring_sizes_[1]));
                sector_idx = static_cast<int>((theta / sector_sizes_[1]));
            } else if (r < min_range_z4_) {
                ring_idx   = static_cast<int>(((r - min_range_z3_) / ring_sizes_[2]));
                sector_idx = static_cast<int>((theta / sector_sizes_[2]));
            } else {
                ring_idx   = static_cast<int>(((r - min_range_z4_) / ring_sizes_[3]));
                sector_idx = static_cast<int>((theta / sector_sizes_[3]));
            }

            czm[ring_idx*16+sector_idx].points.emplace_back(pt);
        }
    }

    for (auto& z : czm) {
        if (z.points.size() > MIN_POINTS) {
            sensor_msgs::PointCloud2 output;
            pcl::toROSMsg(z.points, output);
            pub.publish(output);
        }
    }
}

int main (int argc, char** argv) {
    ros::init (argc, argv, "bumpypatch_node");
    ros::NodeHandle nh;

    pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);

    ros::Subscriber sub = nh.subscribe ("/velodyne_points", 1, cloud_cb);

    ros::spin ();
}
