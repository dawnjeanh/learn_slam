#include <gnuplot-iostream.h>
#include <map>
#include <vector>
#include <cmath>

int main(int argc, char const *argv[])
{
    Gnuplot gp;
    std::vector<std::pair<double, double>> xy_pts_A;
    for (double x = -2; x < 2; x += 0.01)
    {
        double y = x * x * x;
        xy_pts_A.push_back(std::make_pair(x, y));
    }
    std::vector<std::pair<double, double>> xy_pts_B;
    for (double alpha = 0; alpha < 1; alpha += 1.0 / 24.0)
    {
        double theta = alpha * 2.0 * 3.14159;
        xy_pts_B.push_back(std::make_pair(cos(theta), sin(theta)));
    }
    // gp << "set xrange [-2:2]\nset yrange [-2:2]\n";
    gp << "plot" << gp.file1d(xy_pts_A) << "with lines title 'cubic'," << gp.file1d(xy_pts_B) << "with points title 'circle'" << std::endl;
    return 0;
}
