# Covariance Computation
When constructing an optimized portfolio for low-risk and high-return (Max Sharpe, Efficient Frontier), it is important to have the covariance matrix computed, $\Sigma$ in order to be able to work out the variance of the portfolio (the risk) in an effort to minimize it as much as possible. These parameters are related via the formula, $$\sigma_p^2 = \textbf{w}^T \Sigma \textbf{w}$$

Here, $\textbf{w}$ represents the portfolio weighting vector and $\sigma_p^2$ is the portfolio variance which we aim to minimize using a gradient descent based approach. 


Overall, the covariance matrix plays an important part in the later stage of building a low-risk, high return portfolio and is imperative to be computed. 


However, there is an issue, we previously applied different transformations to each cryptocurrencies' dataset in order to ensure it was stationary. For instance, `ETHUSDT` may have had the transformation `[log, diff]` whereas `QTUMUSDT` may have had the transformation `[log, diff, diff]` in order to fulfill the stationarity criteria. This poses an issue, because the covariance is only meaningful when it is taken over the same underlying variable in each of the time series datasets. If one asset has undergone one set of transformations and another asset something different, we have essentially measurede the joint variability of two quite random aspects. This would mean that our resulting portfolio matrix will not correspond to any portfolio risk whatsoever. 


In order to mitigate this issue, we will implement a somewhat approximate method where we compute strictly the log value followed by a differencing operator, `[log, diff]` for all the assets (atop their respective transforms which brought them to stationarity through the ADF and KPSS criterion) and use this particular dataset to compute the covariance matrix. 

**Note: The transforms that brought the time series data for each asset to stationarity will still be used as the raw input for the various strategies for prediction and only the data for computing the covariances is potentially being changed**

In order to implement this idea, we will go back to the `DataProcessor` class and ensure that each transformation done to the data is stored within the CSV file alongside a dedicated `[log, diff]` column for use in the covariance estimation. 

There is another issue when computing the covariance, and that is to do with the range for $t$. Due to the fact that different cryptocurrency assets launched at very different timestamps, we need to find a way to ensure we use aligned data in order to ensure validity in our covariance estimation. The method I have chosen for doing this is to first consider the two datasets we are trying to compute the covariance for and then, by using the `min` and `max` operators, we splice the datsets so that they are both within the same time range and only this time range is used to compute the covariance. This issue solves the problem of inequivalence in $t$ with the only drawback being that for certain covariance estimates, the overlapping time range might be shorter than others but this is unavoidable. 

Now that all the issues have been mitigated, we can go about using the following formula to compute the covariance for two time series datasets: $$\hat{\text{Cov}}(A, B) = \frac{1}{T - 1} \sum_{t = 1}^{T} (r_t^{(A)} - \overline{r}^{(A)})(r_t^{(B)} - \overline{r}^{(B)})$$

Another thing to mention is that we will be using the concept of shrinkage in order to improve our $\Sigma$ estimation. Shrinkage is required in cases where the window size, $T$ is not significantly greater than the number of assets in which case the estimates can get noisy which translates to poor risk estimation. In general, we combat shrinkage with the formula, $$\Sigma_{\text{shrink}} = (1 - \alpha) S + \alpha F$$

Furthermore, we use the $F$ from the Ledoit-Wolf paper as $F = \mu I$, where $\mu = \frac{1}{\text{Num Assets}} \text{trace}(S)$. 

Since we will likely be doing a large number of linear operations, I found it a potentially useful idea to program the covariance computation between two datasets in `C++` and then make it callable in `Python` by building it with `PyBind11`. Furthermore, the covariance matrix, $\Sigma$ is symmetric and thus only half the values need be computed as the otherside can be populated automatically ($\Sigma_{ij} = \Sigma_{ji}$). 

The code which incorporates this is shown below:
```python 
# DataProcessor.py 
...
raw = data['Close'].copy()

# run through your make_stationary to get the ops and final series
stat_series, ops = self.make_stationary(raw)
self.transforms[ticker] = ops

# build all intermediate columns
intermediate = {}
series = raw.copy()
for i, op in enumerate(ops):
    if op == 'log':
        series = np.log(series)
    elif op == 'diff':
        series = series.diff()
    name = "-".join(ops[: i+1])       # e.g. "log", then "log-diff", then "log-diff-diff"
    intermediate[name] = series

# attach them to the DataFrame
for name, ser in intermediate.items():
    data[name] = ser

# ensure we always have a covariance-ready log-diff
if 'log-diff' in intermediate:
    data['Covariance-LogDiff'] = intermediate['log-diff']
else:
    # if your series was already stationary with only log, then diff once:
    data['Covariance-LogDiff'] = np.log(raw).diff()

# finally, store the last intermediate as "Stationary-Close"
last_name = "-".join(ops)
data['Stationary-Close'] = intermediate[last_name]

data.dropna(subset=['Stationary-Close'], inplace=True)
data.drop(columns=['Close'],                         inplace=True)   # remove raw Close
data.rename(columns={'Stationary-Close': 'Close'},    inplace=True)   # now one Close

data = self.compute_technical_indicators_and_lags(data)
data.dropna(inplace=True)
...
```

```cpp
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
```
