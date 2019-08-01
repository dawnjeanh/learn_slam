#ifndef __CONFIG__
#define __CONFIG__

#include "common.hpp"

namespace myslam
{

class Config
{
private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;
    Config(){} // private constructor makes a singleton
public:
    ~Config()
    {
        if (file_.isOpened())
            file_.release();
    }
    // set a new config file
    static void setParameterFile(const std::string &filename)
    {
        if (config_ == nullptr)
            config_ = std::shared_ptr<Config>(new Config);
        config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
        if (config_->file_.isOpened() == false)
        {
            std::cerr << "parameter file" << filename << " does not exist." << std::endl;
            config_->file_.release();
        }
    }
    // access the parameter values
    template<typename T>
    static T get(const std::string &key)
    {
        return T(config_->file_[key]);
    }
};

std::shared_ptr<Config> Config::config_ = nullptr;

} // namespace myslam


#endif // !__CONFIG__