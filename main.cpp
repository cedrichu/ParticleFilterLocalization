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

int num = 30000;

double sig_uv = 0.05;
double sig_r = 0.01;

typedef struct {
    MatrixXd map_mat;
    vector<Vector2d> map_list;
} MapData;

typedef struct {
    MatrixXd samples;
    MatrixXd sigmas;
    VectorXd weights;
    MatrixXd last_weights;
} Particles;

double getLaserScore(int idx, int jdx, double dist, MapData* map_data);

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
    for (int i=0; i<3; ++i){
        odom(i) = stod(ele[i+1]);
    }
    if (last_odom(2) == 999.0) {
        for (int i=0; i<3; ++i){
            motion(i) = 0.0;
        }
    } else {
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

        double ratio = 1.0;

        particles->sigmas(i, 0) = particles->sigmas(i,0) + ratio*sig_uv;
        particles->sigmas(i, 1) = particles->sigmas(i,1) + ratio*sig_r;
    }
}

double getLoss(double& dist, double& rad, Vector3d& sp, MapData* map_data) {
    double dx = cos(rad + sp(2));
    double dy = sin(rad + sp(2));
    double lx = sp(0);
    double ly = sp(1);
    double ray = 0;
    while (lx >= 0 && lx < 800 && ly >=0 && ly < 800) {
        if (map_data->map_mat((int)lx, (int)ly) < 0.1) {
            return abs(ray - dist/10.0);
        } else {
            lx += dx;
            ly += dy;
            ray += 1.0;
        }
    }
    return abs(ray - dist/10.0);
}


void updateWeights(Particles* particles, MapData* map_data, VectorXd& laser, bool& is_resample) {

    int valid_num = 0;
    for(int i = 0; i < num; i++){
        if ((particles->samples(i,0) < 0) || (particles->samples(i,0) >= 800) || (particles->samples(i,1) < 0) || (particles->samples(i,1) >= 800)) {
            particles->weights(i) = 0.0;
        } else if (map_data->map_mat((int)particles->samples(i, 0), (int)particles->samples(i, 1)) != 1.0) {
            particles->weights(i) = 0.0;
        //} else if ( particles->weights(i) < 0.000001/num) {
        //    particles->weights(i) = 0.0;
        } else {
            valid_num += 1;
        }
    }
    
    if (valid_num < 0.95*num)
        is_resample = true;


    int ds_rate = 10;
    int laser_num = 180/ds_rate;
    double rad;
    double loss = 0;

    for(int i = 0; i < num; i++){
        if(particles->weights(i) != 0){
            loss = 0;
            rad = -M_PI/2.0 + M_PI/360.0;
            Vector3d sp;
            sp = particles->samples.row(i);

            for(int j = 0; j < laser_num; j++)
            {
                loss += getLoss(laser(j*ds_rate), rad, sp, map_data);
                rad += (M_PI/180 * ds_rate);
            }
            
            particles->weights(i) = exp(-0.5*loss/pow(25, 2));
        }          
    }

    
    cv::Mat laser_plot = cv::Mat::zeros(500, 500, CV_8UC1);
    for (int j = 0; j < 180; ++j) {
        int idx = laser(j)/10.0 * cos(((double)j)*M_PI/180.0 - M_PI/2.0) + 250;
        int jdx = laser(j)/10.0 * sin(((double)j)*M_PI/180.0 - M_PI/2.0) + 250;
        if (idx >= 0 && idx < 500 && jdx >=0 && jdx < 500) {
            laser_plot.data[idx*500 + jdx] = 255;
        }
    }
    imshow("laser", laser_plot);
    cv::waitKey(10);

    double sum_w = particles->weights.sum();
    particles->weights = particles->weights/(sum_w);
    particles->last_weights = particles->weights;

}

void reSample(Particles* particles, MapData* map_data, bool& is_resample, int time_idx)
{
    if(is_resample == false)
        return;

    double ratio;
    ratio = 1;
    //ratio = 1 + 1000/(double(time_idx*200)+1000);
    int num_w =  num/ratio;
    
    default_random_engine generator;
    VectorXd weights_num;
    weights_num = particles->weights * num_w;


    MatrixXd previous_samples = MatrixXd::Zero(num,3);
    previous_samples = particles->samples;

    std::normal_distribution<double> normal(0.0,1.0);

    double cdf = 0;
    int count_s = 0;
    for (int i = 0; i < num_w; i++){
        cdf = cdf + weights_num(i);
        while (count_s < (int)(cdf) && count_s < num_w) {
            double n_x = normal(generator);
            double n_y = normal(generator);
            double n_theta = normal(generator);
            Vector3d temp_sample;
            temp_sample(0) = previous_samples(i,0) + particles->sigmas(i,0) * n_x; 
            temp_sample(1) = previous_samples(i,1) + particles->sigmas(i,0) * n_y; 
            temp_sample(2) = previous_samples(i,2) + particles->sigmas(i,1) * n_theta; 


            if((temp_sample(0) >= 0) && (temp_sample(0)< 800) && (temp_sample(1) >= 0) && (temp_sample(1) < 800)) {
                if(map_data->map_mat((int)temp_sample(0),(int)temp_sample(1)) == 1.0) {
                    particles->samples(count_s, 0) = temp_sample(0);
                    particles->samples(count_s, 1) = temp_sample(1);
                    particles->samples(count_s, 2) = temp_sample(2);
                    count_s = count_s + 1;
                }
            }
        }
    }
    for (int i = num_w; i < num; i++) {
        int seed = rand()%(map_data->map_list.size());
        particles->samples(i,0) = map_data->map_list[seed](0);
        particles->samples(i,1) = map_data->map_list[seed](1);
        double seed_r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2 * M_PI;
        particles->samples(i,2) = seed_r;
    }
    particles->last_weights = particles->weights;
    particles->weights = VectorXd::Constant(num, 1.0);
    particles->sigmas = MatrixXd::Zero(num,2);
    is_resample = false;
}


void drawParticles(Particles* particles, cv::Mat& output_img) {
    for(int i = 0; i < num; i++) {
        if(particles->last_weights(i) != 0) {
            output_img.data[int(particles->samples(i,0))*800 + int(particles->samples(i,1))] = 127;
        }
    }
}




int main()
{
    MatrixXd map_mat = MatrixXd::Zero(800, 800);
    cv::Mat map_img = cv::Mat::zeros(800, 800, CV_8UC1);
    cv::Mat output_img = cv::Mat::zeros(800, 800, CV_8UC1);  
    ifstream map_file("../data/map/wean.dat");
    parse_map(&map_file, map_mat);
    
    ifstream log_file("../data/log/robotdata1.log");
    //ifstream log_file("../data/log/ascii-robotdata2.log");
    //ifstream log_file("../data/log/ascii-robotdata3.log");
    //ifstream log_file("../data/log/ascii-robotdata4.log");
    //ifstream log_file("../data/log/ascii-robotdata5.log");
    
    string line;

    Vector3d motion;
    Vector3d last_odom(0.0, 0.0, 999.0);
    VectorXd laser = VectorXd::Zero(180, 1);
    MatrixXd samples = MatrixXd::Zero(num, 3);
    vector<Vector2d> map_list;
    MatrixXd sigmas = MatrixXd::Zero(num,2);
    VectorXd weights = VectorXd::Constant(num, 1.0);
    VectorXd last_weights = VectorXd::Constant(num, 1.0);
    bool is_resample = false;
    int time_idx = 0;




    getInitMapList(map_list, map_mat); 
    getInitSamples(samples, map_list);
    setMapImg(map_mat, map_img);

    Particles* particles = new Particles;
    *particles = {samples,sigmas, weights, last_weights};
    MapData* map_data = new MapData;
    *map_data = {map_mat, map_list};

    cv::VideoWriter output;
    output.open ( "outputVideo.avi", CV_FOURCC('M','P','4','2'), 30, cv::Size (800,800), true );
   
    output_img = map_img.clone();
    drawParticles(particles, output_img);
    imshow("map", output_img);
    cv::waitKey(1);

    while (getline(log_file, line)) {
        vector<string> ele;
        split_log(line,' ', ele);
        if (ele[0] == "L") {
            getLocalOdom(ele, last_odom, motion);
            getLaserData(ele, laser);
            addMotion(particles, motion);
            updateWeights(particles, map_data,laser, is_resample);

            //if (time_idx%10 == 0) {
                is_resample = true;
            //}

            reSample(particles, map_data, is_resample, time_idx);

            output_img = map_img.clone();
            drawParticles(particles, output_img);
            imshow("map", output_img);
            
            output.write ( output_img );
            
            if(cv::waitKey(50) >= 0) break;

        }
        else if (ele[0] == "O")
        {
            //getLocalOdom(ele, last_odom, motion);
            //addMotion(particles, motion);
        }
        time_idx += 1;
    }
    output.release();

    return 0;
}



