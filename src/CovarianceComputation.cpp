// CovarianceComputation.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <numeric>
#include <vector>
#include <map>
#include <string>
#include <set>

namespace py = pybind11;

// compute sample covariance between two vectors
double sample_cov(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double c = 0.0;
    for(int i = 0; i < n; ++i)
        c += (x[i] - mean_x) * (y[i] - mean_y);
    return c / (n - 1);
}

// shrink S toward μ I: Σ_shrink = (1-α) S + α μ I
std::vector<std::vector<double>> ledoit_wolf_shrink(
    const std::vector<std::vector<double>>& S,
    double alpha
) {
    int p = S.size();
    // compute μ = trace(S) / p
    double trace = 0.0;
    for(int i = 0; i < p; ++i)
        trace += S[i][i];
    double mu = trace / p;

    // build Σ_shrink
    std::vector<std::vector<double>> Shr(p, std::vector<double>(p));
    for(int i = 0; i < p; ++i) {
        for(int j = 0; j < p; ++j) {
            double target = (i==j ? mu : 0.0);
            Shr[i][j] = (1.0 - alpha) * S[i][j] + alpha * target;
        }
    }
    return Shr;
}

// data_dict: ticker -> DataFrame with ['Open Time','Covariance-LogDiff']
py::dict compute_covariance(py::dict data_dict, double shrinkage=0.1) {
    // 1) Collect tickers
    std::vector<std::string> tickers;
    for(auto item : data_dict)
        tickers.push_back(py::cast<std::string>(item.first));
    int p = tickers.size();

    // 2) Build time series maps and find common timestamps
    std::map<std::string,std::map<std::string,double>> series_map;
    std::set<std::string> common_times;
    bool first = true;

    for(auto &t : tickers) {
        py::object df = data_dict[t.c_str()];
        // get columns as vectors
        auto times = df.attr("__getitem__")("Open Time")
                       .attr("astype")("str")
                       .cast<std::vector<std::string>>();
        auto vals  = df.attr("__getitem__")("Covariance-LogDiff")
                       .cast<std::vector<double>>();

        // map time -> value
        std::map<std::string,double> m;
        for(size_t i=0; i<times.size(); ++i)
            m[times[i]] = vals[i];
        series_map[t] = m;

        if(first) {
            common_times.insert(times.begin(), times.end());
            first = false;
        } else {
            std::set<std::string> tmp;
            for(auto &tm: common_times)
                if(m.count(tm)) tmp.insert(tm);
            common_times.swap(tmp);
        }
    }

    // 3) Extract common samples into matrix X[p][T]
    int T = common_times.size();
    std::vector<std::string> times(common_times.begin(), common_times.end());
    std::vector<std::vector<double>> X(p, std::vector<double>(T));
    for(int i=0; i<p; ++i) {
        auto &m = series_map[tickers[i]];
        for(int j=0; j<T; ++j)
            X[i][j] = m[times[j]];
    }

    // 4) Compute sample covariance matrix S
    std::vector<std::vector<double>> S(p, std::vector<double>(p));
    for(int i = 0; i < p; ++i) {
        auto &map_i = series_map[tickers[i]];
        for(int j = i; j < p; ++j) {
            auto &map_j = series_map[tickers[j]];
            // 1) find pairwise common timestamps
            std::vector<double> xi, yj;
            for (auto &kv : map_i) {
                const auto &t = kv.first;
                auto it = map_j.find(t);
                if (it != map_j.end()) {
                    xi.push_back(kv.second);
                    yj.push_back(it->second);
                }
            }
            // 2) compute covariance if we have enough points
            double c = (xi.size()>1)
                       ? sample_cov(xi, yj)
                       : 0.0;
            S[i][j] = S[j][i] = c;
        }
    }

    // 5) Apply Ledoit-Wolf shrinkage
    auto Sshrunk = ledoit_wolf_shrink(S, shrinkage);

    // 6) Build return dict of dicts
    py::dict result;
    for(int i=0;i<p;++i) {
        py::dict row;
        for(int j=0;j<p;++j) {
            // convert std::string to const char* so operator[] is happy
            row[tickers[j].c_str()] = Sshrunk[i][j];
        }
        result[tickers[i].c_str()] = row;
        }
    return result;
}

PYBIND11_MODULE(covcomp, m) {
    m.doc() = "Covariance computation with Ledoit-Wolf shrinkage";
    m.def("compute_covariance", &compute_covariance,
          py::arg("data_dict"), py::arg("shrinkage") = 0.1,
          "Compute shrinkage-adjusted covariance matrix");
}