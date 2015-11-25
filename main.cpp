#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
#define MAP_x 800
#define MAP_y 800

int num = 500;

double sig_uv = 0.5;
double sig_r = 0.1;

typedef struct {
    MatrixXd map_mat;
    vector<Vector2d> map_list;
} MapData;

typedef struct {
    MatrixXd samples;
    MatrixXd sigmas;
    VectorXd weights;
} Particles;


void split_map(const string& s, char c, MatrixXd& map, const int& idx) 
{
    string::size_type i = 0;
    string::size_type j = s.find(c);
    vector<string> v;

    while (j != string::npos) 
    {
        v.push_back(s.substr(i, j-i));
        i = ++j;
        j = s.find(c, j);

        if (j == string::npos)
            v.push_back(s.substr(i, s.length()));
    }    
    for(int i = 0; i < MAP_x; i++)
        map(idx, i) = stod(v[i]);
}

void split_log(const string& s, char c, vector<string>& v) 
{
    string::size_type i = 0;
    string::size_type j = s.find(c);


    while (j != string::npos) 
    {
        v.push_back(s.substr(i, j-i));
        i = ++j;
        j = s.find(c, j);

        if (j == string::npos)
            v.push_back(s.substr(i, s.length()));
    }

}

void parse_map(ifstream* file, MatrixXd& map)
{
    string line;
    int line_num = 0;
    int skip_num = 7;
    char delimiter = ' ';
    int i = 0;

    while (getline(*file, line))
    {
        if( line_num >= skip_num) 
        {
            split_map(line, delimiter, map, i);
            i++;
        }	
        line_num++;
    }
}

void setMapImg(MatrixXd& map_mat, cv::Mat& map_img) {
    for (int i=0; i< 800; i++) {
        for (int j=0; j<800; ++j) {
            if (map_mat(i, j) == 0) {
                map_img.data[i*800 + j] = 0;
            } else if (map_mat(i, j) == 1) {
                map_img.data[i*800 + j] = 255;
            }
            else {
                map_img.data[i*800 + j] = map_mat(i, j) * 255;
            }
        }
    }
}
void getLocalOdom(vector<string>& ele, Vector3d& last_odom, Vector3d& motion) {
    Vector3d odom;
    if (last_odom(2) == 999) {
        for (int i=0; i<3; ++i){
            motion(i) = 0.0;
        }
    } else {

        for (int i=0; i<3; ++i){
            odom(i) = stod(ele[i+1]);
        }
        //get motion
        double d_x = odom(0) - last_odom(0);
        double d_y = odom(1) - last_odom(1);
        motion(2) = odom(2) - last_odom(2);
        motion(0) = (d_x*cos(-odom(2)) - d_y*sin(-odom(2)))/10.0;
        motion(1) = (d_x*sin(-odom(2)) + d_y*cos(-odom(2)))/10.0;
    }
    //set last_odom
    for (int i=0; i<3; ++i){
        last_odom(i) = odom(i);
    }
}

void getLaserData(vector<string>& ele, VectorXd& laser) {
    for(int i = 0; i < 180; ++i){
         laser(i) = stod(ele[i+7]);
    }
}

void getInitMapList(vector<Vector2d>& list, MatrixXd& map_mat)
{
    for(int i = 0; i < 800; i++)
        for(int j = 0; j < 800; j++)
            if(map_mat(i,j) == 1){
                Vector2d coord((double(i)), (double(j)));
                list.push_back(coord);
            }

}

void getInitSamples(MatrixXd& samples, vector<Vector2d>& list ) {
    for (int i = 0; i < num; ++i) {
        int seed = rand()%list.size();
        samples(i,0) = list[seed](0);
        samples(i,1) = list[seed](1);
        double seed_r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2 * M_PI;
        samples(i,2) = seed_r;
    }
} 
void addMotion(Particles* particles, Vector3d motion)
{

    for(int i = 0; i < num; i++)
    {
        double d_x = motion(0)*cos(particles->samples(i,2)) - motion(1)*sin(particles->samples(i,2)); 
        double d_y = motion(0)*sin(particles->samples(i,2)) + motion(1)*cos(particles->samples(i,2));
        particles->samples(i,0) = particles->samples(i,0) + d_x;
        particles->samples(i,1) = particles->samples(i,1) + d_y;
        particles->samples(i,2) = particles->samples(i,2) + motion(2);

        double ratio = sqrt(motion(0)*motion(0) + motion(1)*motion(1));

        particles->sigmas(i, 0) = particles->sigmas(i,0) + ratio*sig_uv;
        particles->sigmas(i, 1) = particles->sigmas(i,1) + ratio*sig_r;
    }
}

void updateWeights(Particles* particles, MapData* map_data, VectorXd& laser, bool& is_resample)
{

    int valid_num = 0;
    is_resample = 0;
    for(int i = 0; i < num; i++){
        if ((particles->samples(i,0) < 0) || (particles->samples(i,0) >= 800) || (particles->samples(i,1) < 0) || (particles->samples(i,1) >= 800))
            particles->weights(i) = 0;
        else if(map_data->map_mat(particles->samples(i, 0), particles->samples(i, 1)) != 1)
            particles->weights(i) = 0;
        else if ( particles->weights(i) < 0.01/num)
            particles->weights(i) = 0;
        else
            valid_num += 1;
    }
    

    particles->weights = particles->weights;///(double(valid_num)); //possible NAN
    if (valid_num < 0.95*num)
        is_resample = 1;


    int ds_rate = 10;
    int laser_num = 180/ds_rate;
    int count_match = 0;
    MatrixXd l_data = MatrixXd::Zero(laser_num,2);
    double rad;

    for(int i = 0; i < num; i++){
        if(particles->weights(i) != 0){
            l_data = MatrixXd::Zero(laser_num,2);
            rad = -M_PI/2 + M_PI/360;
            for(int j = 0; j < laser_num; j++)
            {
                l_data(j,0) = laser(j*ds_rate)/10 * cos(rad + particles->samples(i,2)) + particles->samples(i,0);//laser's index??
                l_data(j,1) = laser(j*ds_rate)/10 * sin(rad + particles->samples(i,2)) + particles->samples(i,1);
                rad += (M_PI/180 * ds_rate);
            }


            count_match = 0;
            for(int k = 0; k < laser_num; k++)
                if((l_data(k,0) >= 0) && (l_data(k,0)< 800) && (l_data(k,1) >= 0) && (l_data(k,1) < 800))
                    if(map_data->map_mat(l_data(k,0), l_data(k,1)) == 0 )//need better function()
                        count_match = count_match + 1;

            particles->weights(i) = particles->weights(i) * count_match;
        }          
    }
       
    int sum_w = particles->weights.sum();
    particles->weights = particles->weights/(double(sum_w));
}


void reSample(Particles* particles, MapData* map_data, bool& is_resample, int time_idx)
{
    if(is_resample == 0)
        return;


    default_random_engine generator;
    VectorXd weights_num = particles->weights;
    int ratio = 1 + 10/(time_idx+10);
    weights_num = weights_num * num/ratio;
    vector<double> weights_vector(weights_num.data(), weights_num.data() + weights_num.size());
    discrete_distribution<int> distribution (weights_vector.begin(), weights_vector.end());
    MatrixXd previous_samples = particles->samples;
    
    for (int i = 0; i < num/ratio; i++){
        int number = distribution(generator);
        Vector3d temp_sample;
        temp_sample(0) = previous_samples(number,0) + particles->sigmas(number,0) ; 
        temp_sample(1) = previous_samples(number,1) + particles->sigmas(number,0) ; 
        temp_sample(2) = previous_samples(number,2) + particles->sigmas(number,1) ; 
        if((temp_sample(0) >= 0) && (temp_sample(0)< 800) && (temp_sample(1) >= 0) && (temp_sample(1) < 800))
            if(map_data->map_mat(temp_sample(0),temp_sample(1)) == 1)
            {
                particles->samples(0) = temp_sample(0);
                particles->samples(1) = temp_sample(1);
                particles->samples(2) = temp_sample(2);
            }

    }
   

    for (int i = num/ratio; i < num; i++) {
        int seed = rand()%(map_data->map_list.size());
        particles->samples(i,0) = map_data->map_list[seed](0);
        particles->samples(i,1) = map_data->map_list[seed](1);
        double seed_r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2 * M_PI;
        particles->samples(i,2) = seed_r;
    }

    particles->weights = VectorXd::Constant(num, 1.0);
    particles->sigmas = MatrixXd::Zero(num,2);
    is_resample = 0;
}




int main()
{
    MatrixXd map_mat = MatrixXd::Zero(800, 800);
    cv::Mat map_img = cv::Mat::zeros(800, 800, CV_8UC1);  
    ifstream map_file("../data/map/wean.dat");
    parse_map(&map_file, map_mat);
    ifstream log_file("../data/log/robotdata1.log");
    string line;

    Vector3d motion;
    Vector3d last_odom(0.0, 0.0, 999.0);
    VectorXd laser = VectorXd::Zero(180, 1);
    MatrixXd samples = MatrixXd::Zero(num, 3);
    vector<Vector2d> map_list;
    MatrixXd sigmas = MatrixXd::Zero(num,2);
    VectorXd weights = VectorXd::Constant(num, 1.0);
    bool is_resample = 0;
    int time_idx = 0;

    

    //weights = weights/(double(num));
    
    getInitMapList(map_list, map_mat); 
    getInitSamples(samples, map_list);
    setMapImg(map_mat, map_img);

    Particles* particles = new Particles;
    *particles = {samples,sigmas, weights};
    MapData* map_data = new MapData;
    *map_data = {map_mat, map_list};

    //imshow("map", map_img);
    //cv::waitKey(0);

    while (getline(log_file, line)) {
        vector<string> ele;
        split_log(line,' ', ele);
        if (ele[0] == "L") {
            getLocalOdom(ele, last_odom, motion);
            getLaserData(ele, laser);
            addMotion(particles, motion);
            updateWeights(particles, map_data,laser, is_resample);
            reSample(particles, map_data, is_resample, time_idx);
        }
        time_idx += 1;
    }

    return 0;
}



