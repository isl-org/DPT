#include <torch/script.h> // One-stop header.
#include <torch/nn/functional/upsampling.h>

#include <iostream>
#include <memory>
#include <string> 
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define GPU 1 

template<class T>
T remove_extension(T const & filename)
{
  typename T::size_type const p(filename.find_last_of('.'));
  return p > 0 && p != T::npos ? filename.substr(0, p) : filename;
}
std::string base_name(std::string const & path)
{
  return path.substr(path.find_last_of("/\\") + 1);
}
int main(int argc, const char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    std::cout<<"Module loaded successfuly"<<std::endl; 
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  module.to(at::kCUDA);

  // module.to(at::kHalf);
  // std::vector<torch::jit::IValue> inputs; 
  // auto tensor_in = torch::ones({1, 3, 32, 96}); 
  // if(GPU){
  // module.to(at::kCUDA);
  // tensor_in.to(at::kCUDA);
  // }
  // inputs.push_back(tensor_in);
  // double t = (double)cv::getTickCount(); 
  

  std::vector<cv::String> fn;
  glob(argv[3], fn, false);

  std::vector<cv::Mat> images;
  size_t count = fn.size(); //number of png files in images folder
  for (size_t i=0; i<count; i++){
    cv::Mat data;
    data = cv::imread(fn[i], 1);
    std::cout<<"Reading image: "<<remove_extension(base_name(fn[i]))<<std::endl; 
    int col = data.cols; 
    int row = data.rows; 
    cv::cvtColor(data, data, CV_BGR2RGB);
    int64 start = cv::getTickCount();
    cv::resize(data, data, cv::Size(672, 384), cv::INTER_CUBIC);

    data.convertTo(data, CV_32FC3);

    cv::Mat ch_first = data.clone();
    // cv::imshow("Display window", ch_first);
    // cv::waitKey(0);
    std::cout<<data.type()<<std::endl;
    if (data.type() != CV_32FC3) std::cout << "wrong type" << std::endl;

    float* feed_data = (float*)data.data;
    float* ch_first_data = (float*)ch_first.data;

    for (int p = 0; p < (int)data.total(); ++p)
    {
      // R
      ch_first_data[p] = feed_data[p * 3];
      // G
      ch_first_data[p + (int)data.total()] = feed_data[p * 3 + 1];
      // B
    // cv::imshow("Display window", data);
      ch_first_data[p + 2 * (int)data.total()] = feed_data[p * 3 + 2];
    }

    // torch::Tensor mean = torch::tensor({ 0.5f, 0.5f, 0.5f }).toType(torch::kFloat16).to(torch::Device("cuda")); 
    

    torch::Tensor image_input = torch::from_blob((float*)ch_first.data, { 1, 3, data.rows, data.cols});
    image_input = image_input.to(at::kCUDA);
    image_input = image_input.div(255);
    image_input = (image_input.sub(0.5f)).div(0.5f);
    image_input = image_input.toType(torch::kFloat32);


    // image_input = image_input.to(at::kCUDA, torch::kHalf);

    // image_input.to();

    auto net_out = module.forward({ image_input }).toTensor();
    // net_out = net_out[0]; 
    namespace F = torch::nn::functional;
    std::cout<<"#############SHIT1"<<std::endl; 

    net_out = F::interpolate(
          net_out.unsqueeze(1),
          F::InterpolateFuncOptions()
                  .mode(torch::kBicubic)
                  .size(std::vector<int64_t>({row, col})).align_corners(false)); 
    std::cout<<"#############SHIT2"<<std::endl; 

    net_out = net_out.squeeze();
    net_out = (net_out.min(net_out.max()) / (net_out.max() - net_out.min())).mul(65535.0f); 
    // net_out = net_out.to(torch::kFloat32);
    // net_out = net_out.mul(255);
    net_out = net_out.to(torch::kCPU);
  
    // net_out = net_out.permute({ 1, 2, 0 }).detach();
    // net_out = torch::softmax(net_out, 2);
    // net_out = net_out.argmax(2);
    // net_out = net_out.mul(255).clamp(0, 255).to(torch::kU8);
    // net_out = net_out.to(torch::kCPU);

    int height = net_out.sizes()[0];
    int width = net_out.sizes()[1]; 
    int x = net_out.sizes()[2];

    std::cout<<"X: "<<x<<std::endl;
    std::cout<<"height: "<<height<<std::endl;
    std::cout<<"width: "<<width<<std::endl; 
    std::cout << "DONE!!!"<<std::endl;
    // net_out.to(torch::kByte);
    
    try
    {
      cv::Mat output_mat(cv::Size{width, height}, CV_32FC1, net_out.data_ptr());
      output_mat.convertTo(output_mat, CV_16UC1); 
      double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
      std::cout << "FPS : " << fps << std::endl;
      // output_mat = output_mat.mul(1 / 255);
      // cv::imshow("Display window", output_mat);
      // cv::waitKey(0);
      int rows = output_mat.rows;
      int cols = output_mat.cols; 
      std::cout<<"rows: "<<rows<<std::endl;
      std::cout<<"cols: "<<cols<<std::endl;
      // cv::imshow("some.jpg", output_mat);
      std::string extention = ".png";
      std::string output_address = "./output/"; 
      output_address.append(remove_extension(base_name(fn[i])).append(extention)); 
      cv::imwrite(output_address, output_mat);
      
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }

    // cv::cvtColor(data, data, CV_BGR2RGB);
    // cv::resize(data, data, cv::Size(640, 384));

    // torch::Tensor imgTensor = torch::from_blob(data.data, {data.rows, data.cCV_64FC3
    // cv::imshow("Display window", data);
    // int k = cv::waitKey(0); // Wait for a keystroke in the window
    
    // cv::Mat ch_first = data.clone();
    // using namespace std; 
    
    // if (data.type() != CV_32FC3) cout << "wrong type" << endl;
    
    // float* feed_data = (float*)data.data;
    // std::cout<<"HERE3"; 

    // float* ch_first_data = (float*)ch_first.data;
    // std::cout<<"HERE2"; 

    // for (int p = 0; p < (int)data.total(); ++p)
    // {
    // 	// R
    // 	ch_first_data[p] = feed_data[p * 3];
    // 	// G
    // 	ch_first_data[p + (int)data.total()] = feed_data[p * 3 + 1];
    // 	// B
    // 	ch_first_data[p + 2 * (int)data.total()] = feed_data[p * 3 + 2];
    // }
    // std::cout<<"HERE1"; 
    // torch::Tensor image_input = torch::from_blob((float*)ch_first.data, { 1, data.rows, data.cols, 3 });
    // image_input = image_input.toType(torch::kFloat16);
    // cout<<"Matrix: "<<ty.c_str(); 
  }
  // cv::imshow("Display window", data);
  // cv::waitKey(0);


}

