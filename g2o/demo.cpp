#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <gnuplot-iostream.h>

using Block = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;

class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl()
    {
        _estimate << 0, 0, 0;
    }
    virtual void oplusImpl(const double *update)
    {
        _estimate += Eigen::Vector3d(update);
    }
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
};

class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
private:
    double _x;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x): BaseUnaryEdge(), _x(x) {}
    void computeError()
    {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
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
    //构建图
    std::unique_ptr<Block::LinearSolverType> linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<Block>(std::move(linearSolver))); //梯度下降
    // g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    // g2o::OptimizationAlgorithmDogleg *solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
    g2o::SparseOptimizer optimizer;//图模型
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    //添加顶点
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));
    v->setId(0);
    optimizer.addVertex(v);
    //添加边
    for (int i = 0; i < N; ++i)
    {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 /(1.0 * 1.0));
        optimizer.addEdge(edge);
    }
    //计算
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    Eigen::Vector3d abc_estimate = v->estimate();
    std::cout << "abc=" << abc_estimate.transpose() << std::endl;
    //显示
    Gnuplot gp;
    std::vector<std::pair<double, double>> xy_pts_data, xy_pts_data_real, xy_pts_data_fit;
    for (int i = 0; i < N; ++i)
    {
        xy_pts_data.push_back(std::make_pair(x_data[i], y_data[i]));
        xy_pts_data_real.push_back(std::make_pair(x_data[i], y_data_real[i]));
        // xy_pts_data_fit.push_back(std::make_pair(x_data[i], y_data_fit[i]));
    }
    gp << "plot"
       << gp.file1d(xy_pts_data) << "with points pt 7 title 'data',"
       << gp.file1d(xy_pts_data_real) << "with line lt 2 title 'real',"
       /* << gp.file1d(xy_pts_data_fit) << "with line lt 1 title 'fit'" */ << std::endl;

    return 0;
}