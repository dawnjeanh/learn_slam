#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include <ceres/ceres.h>
#include <gnuplot-iostream.h>

//代价函数计算模型
struct CURVE_FITTING_COST
{
    const double _x, _y;
    CURVE_FITTING_COST(double x, double y): _x(x), _y(y){}
    //残差
    template <typename T>
    bool operator() (const T* const abc, T* residual) const
    {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }
};


int main(int argc, char const *argv[])
{
    double a = 1.0, b = 2.0, c = 1.0;
    int N = 100;
    double abc[3] = {0};
    //数据
    std::vector<double> x_data, y_data, y_data_real;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < N; ++i)
    {
        double x = i / 100.0;
        double rnd = distribution(generator);
        x_data.push_back(x);
        y_data.push_back(std::exp(a * x * x + b * x + c) + rnd);
        y_data_real.push_back(std::exp(a * x * x + b * x + c));
    }
    //最小二乘
    ceres::Problem problem;
    for (int i = 0; i < N; ++i)
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])),
            nullptr,
            abc
        );
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "summary:" << std::endl << summary.BriefReport() << std::endl;
    for (auto a : abc)
    {
        std::cout << a << " ";
    }
    std::cout << std::endl;
    //拟合数据
    std::vector<double> y_data_fit;
    for (int i = 0; i < N; ++i)
    {
        double x = i / 100.0;
        y_data_fit.push_back(std::exp(abc[0] * x * x + abc[1] * x + abc[2]));
    }
    //显示
    Gnuplot gp;
    std::vector<std::pair<double, double>> xy_pts_data, xy_pts_data_real, xy_pts_data_fit;
    for (int i = 0; i < N; ++i)
    {
        xy_pts_data.push_back(std::make_pair(x_data[i], y_data[i]));
        xy_pts_data_real.push_back(std::make_pair(x_data[i], y_data_real[i]));
        xy_pts_data_fit.push_back(std::make_pair(x_data[i], y_data_fit[i]));
    }
    gp << "plot"
       << gp.file1d(xy_pts_data) << "with points pt 7 title 'data',"
       << gp.file1d(xy_pts_data_real) << "with line lt 2 title 'real',"
       << gp.file1d(xy_pts_data_fit) << "with line lt 1 title 'fit'" << std::endl;

    return 0;
}
