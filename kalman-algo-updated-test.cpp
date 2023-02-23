/*
 * Original implementation in CUDA C can be found in
 * https://github.com/rapidsai/cuml/blob/branch-22.12/cpp/src/arima/batched_kalman.cu
 * Requires nvc++ ver 22.11
 * Requires GNU gcc version 10.2 (for C++20 headers)
 * Compile with:
 * nvc++ -O3 -std=c++23 -stdpar=gpu -gpu=cc86 -gpu=managed -gpu=fma -gpu=fastmath -gpu=autocollapse -gpu=loadcache:L2 -gpu=unroll -o gpu-kalman-algo-test.exe kalman-valgo-test.cpp
 * nvc++ -O3 -std=c++23 -stdpar=multicore -o multicore-kalman-algo-test.exe kalman-algo-test.cpp
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <random>

#include <concepts> 
#include <ranges>
#include <type_traits>

#include <algorithm>
#include <vector>
#include <array>
#include <memory>
#include <numeric>
#include <execution>

#include <experimental/mdspan>

namespace stdex = std::experimental;

#ifndef BSIZE
#define BSIZE 32
#endif

#ifndef NOBS
#define NOBS 8
#endif

#ifndef FCSTEPS
#define FCSTEPS 8
#endif

#ifdef DOUBLE_PRECISION
using Float    = double;
using ReduceTp = Float;
#else
using Float    = float;
using ReduceTp = Float;
#endif

// Simple arithmetic type
template <typename T>
concept ArithmeticTp = requires{
  requires std::is_floating_point_v<T>;
};

// Generic container type:
template <typename T>
concept GenericContainerTp  = requires (T t) {
  t.begin();
  t.end();
  t.data();
  t.size();
};

constexpr int nObs     = NOBS;
constexpr int fc_steps = FCSTEPS;

constexpr int bSize = BSIZE;
constexpr int D     = 8;

using vFloatN  = std::vector<std::array<Float, bSize*D >>;
using vFloatNN = std::vector<std::array<Float, bSize*D*D>>;
using vFloat1  = std::vector<std::array<Float, 1>>;

//
using indx_type     = int; 
using dyn_indx_type = size_t; 

// Static objects views: 
using Left2DView      =  stdex::mdspan<Float, stdex::extents<indx_type, D, D>, stdex::layout_left, stdex::default_accessor<Float>>;
using Left2DCView     =  stdex::mdspan<const Float, stdex::extents<indx_type, D, D>, stdex::layout_left, stdex::default_accessor<const Float>>;

using Right2DView     =  stdex::mdspan<Float, stdex::extents<indx_type, D, D>, stdex::layout_right, stdex::default_accessor<Float>>;
using Right2DCView    =  stdex::mdspan<const Float, stdex::extents<indx_type, D, D>, stdex::layout_right, stdex::default_accessor<const Float>>;

using Left1DView      =  stdex::mdspan<Float, stdex::extents<indx_type, D>, stdex::layout_left, stdex::default_accessor<Float>>;
using Left1DCView     =  stdex::mdspan<const Float, stdex::extents<indx_type, D>, stdex::layout_left, stdex::default_accessor<const Float>>;

//! Thread-local Matrix-Vector multiplication.
template <typename out_vec_t, typename in_mat_t, typename in_vec_t,  bool is_scaled = false>
inline void MV(out_vec_t out, const in_mat_t A, const in_vec_t v, const Float alpha = 0.f)
{
  constexpr int A_range_0 = A.extent(0); 
  constexpr int A_range_1 = A.extent(1);   
#pragma unroll
  for (int i = 0; i < A_range_0; i++) {
    ReduceTp sum = A(i,0) * v(0);
#pragma unroll    
    for (int j = 1; j < A_range_1; j++) {
      sum += A(i,j) * v(j);
    }
    out(i) = is_scaled ? alpha*sum : sum;
  }
}


//! Thread-local Matrix-Matrix multiplication.
template <typename out_mat_t, typename in_mat1_t, typename in_mat2_t>
inline void MM(out_mat_t out, const in_mat1_t A, const in_mat2_t B)
{
#pragma unroll
  for (int i = 0; i < A.extent(0); i++) {
#pragma unroll  
    for (int j = 0; j < B.extent(0); j++) {
      ReduceTp sum = A(i,0) * B(j,0);
#pragma unroll      
      for (int k = 1; k < B.extent(1); k++) {
        sum += A(i, k) * B(j, k);
      }
      out(i,j) = sum;
    }
  }
}

template<int d> 
inline decltype(auto) 
kalman_update(auto &a, 
	      auto &vs_,
	      auto &t,  
	      auto &p, 
	      auto &Fs_, 
	      const auto &ys_,
	      const auto &rqr, 
	      const auto &z, 
	      const Float mu, 
	      const int n_diff,
	      const int nseries,	       
	      const int nobs){
  
  std::array<Float, d>  k_;
  Left1DView k(k_.data()); 
  //
  std::array<Float, d*d> tp_;
  Left2DView tp(tp_.data());   
  //  
  std::array<Float, d*d> l_tmp_;
  Left2DView l_tmp(l_tmp_.data());     
    
  Float b_sum_logFs = 0.0;
    
  for (int it = 0; it < nobs; it++) {
  // 1. v = y - Z*alpha
    Float vs_it = ys_[it];//strided access with stride nseries
    if (n_diff == 0) {
      vs_it -= a(0);
    } else {
#pragma unroll      
      for (int i = 0; i < a.extent(0); i++) {
        vs_it -= a(i) * z(i);
      }
    }
    vs_(it) = vs_it;

    // 2. F = Z*P*Z'
    Float _Fs = n_diff == 0 ? p(0,0) : 0.0;
      
    if (n_diff != 0){
#pragma unroll        
      for (int i = 0; i < p.extent(0); i++) {
#pragma unroll           
        for (int j = 0; j < p.extent(1); j++) {
          _Fs += p(i, j) * z(i) * z(j);
        }
      }
    }
      
    Fs_(it) = _Fs;//strided access with stride nseries
    if (it >= n_diff) b_sum_logFs += log(_Fs);

    // 3. K = 1/Fs[it] * T*P*Z'
    // TP = T*P
    MM(tp, t, p);
    // K = 1/Fs[it] * TP*Z'
    Float _1_Fs = 1.0 / _Fs;
    if (n_diff == 0) {
#pragma unroll      
      for (int i = 0; i < k.extent(0); i++) {
        k(i) = _1_Fs * tp(i, 0);
      }
    } else {
      MV<decltype(k), decltype(tp), decltype(z), true>(k, tp, z, _1_Fs);
    }

    // 4. alpha = T*alpha + K*vs[it] + c
    // tmp = T*alpha
    auto l_tmp_col = stdex::submdspan(l_tmp, std::experimental::full_extent, 0);
    MV(l_tmp_col, t, a);
    // alpha = tmp + K*vs[it]
#pragma unroll      
    for (int i = 0; i < a.extent(0); i++) {
      a(i) = l_tmp_col(i) + k(i) * vs_it;
    }
    // alpha = alpha + c
    a(n_diff) += mu;

    // 5. L = T - K * Z
    // L = T (L is tmp)
#pragma unroll      
    for (int i = 0; i < l_tmp.extent(0); i++) {
#pragma unroll    
      for (int j = 0; j < l_tmp.extent(1); j++) {
        l_tmp(i, j) = t(i,j);
      }
    }
    // L = L - K * Z
    if (n_diff == 0) {
#pragma unroll      
      for (int i = 0; i < l_tmp.extent(0); i++) {
        l_tmp(i,0) -= k(i);
      }
    } else {
#pragma unroll      
      for (int i = 0; i < l_tmp.extent(0); i++) {
#pragma unroll        
        for (int j = 0; j < l_tmp.extent(1); j++) {
          l_tmp(i,j) -= k(i) * z(j);
        }
      }
    }

    // 6. P = T*P*L' + R*Q*R'
    // P = TP*L'
    Right2DCView l_tmp_transp(l_tmp_.data());
    //    
    MM(p, tp, l_tmp_transp);
    // P = P + RQR
#pragma unroll      
    for (int i = 0; i < p.extent(0); i++) {
#pragma unroll    
      for (int j = 0; j < p.extent(1); j++) {
        p(i,j) += rqr(i,j);
      }
    }
  }
    
  return b_sum_logFs;
} 

template<int d, bool conf_int> 
inline void 
kalman_forecast(auto &a, 
		auto &fc_,
	        auto &t,  
	        auto &p,
		auto &F_fc_,
	        const auto &rqr, 
	        const auto &z, 
	        const Float mu, 
	        const int n_diff,
	        const int nseries, 
	        const int fc_steps){
  
  std::array<Float, d*d> tp_;
  Left2DView tp(tp_.data());   
  //  
  std::array<Float, d*d> l_tmp_;
  Left2DView l_tmp(l_tmp_.data());  
   
  for (int it = 0; it < fc_steps; it++) {

    if (n_diff == 0) {
      fc_(it) = a(0);
    } else {
      Float pred = a(0) * z(0);
#pragma unroll        
      for (int i = 1; i < a.extent(0); i++) {
        pred += a(i) * z(i);
      }
      fc_(it) = pred;
    }

    // alpha = T*alpha + c
    auto l_tmp_col = stdex::submdspan(l_tmp, std::experimental::full_extent, 0);
    MV(l_tmp_col, t, a);
#pragma unroll      
    for (int i = 0; i < a.extent(0); i++) {
      a(i) = l_tmp_col(i);
    }
    a[n_diff] += mu;

    if (conf_int) {
      if (n_diff == 0) {
        F_fc_(it) = p(0,0);
      } else {
        Float _Fs = 0.0;
#pragma unroll          
        for (int i = 0; i < p.extent(0); i++) {
#pragma unroll          
          for (int j = 0; j < p.extent(1); j++) {
            _Fs += p(i , j) * z(i) * z(j);
          }
        }
        F_fc_(it) = _Fs;
      }

      // P = T*P*T' + RR'
      // TP = T*P
      MM(tp, t, p);
      // P = TP*T'
      Right2DCView t_transp(t.data_handle());    
      MM(p, tp, t_transp);      
      // P = P + RR'
#pragma unroll      
      for (int i = 0; i < p.extent(0); i++) {
#pragma unroll    
        for (int j = 0; j < p.extent(1); j++) {
          p(i,j) += rqr(i,j);
        }
      }
    }
  }//end for
}    

//
constexpr bool intercept = true;
constexpr bool forecast  = true;
//
void dispatch_kalman_kernel(auto&& kalman_alg, const int nseries){
  //	
  auto policy = std::execution::par_unseq;
  //
  auto outer_loop_range = std::ranges::views::iota(0, nseries);
  //
  std::for_each(policy,
                std::ranges::begin(outer_loop_range),
                std::ranges::end(outer_loop_range),
                kalman_alg);
                
  return;	
}

template <GenericContainerTp container_tp>
class GenericContainerWrapper{
  public:	
    using data_tp = typename container_tp::value_type;	
    //
    container_tp data_container;	

    GenericContainerWrapper(const size_t n) : data_container(n) { }

    GenericContainerWrapper(const size_t n, std::default_random_engine &g, std::uniform_real_distribution<Float> &d) : data_container(n) {
      for (auto &v : data_container) {
        if constexpr (not std::is_arithmetic<decltype(v)>::value) {
          for (auto &u : v) u = d(g);
	}
      }    	  
    }

    GenericContainerWrapper(const container_tp &src) : data_container(src)  {}
    //
    auto Get() { 
      return GenericContainerWrapper<std::span<data_tp>>( std::span{data_container} ); 
    }
    //
    template<int ...d>
    auto Accessor(const size_t tid, const size_t batch_id,  std::array<int, sizeof...(d)> strides) {
       using View = stdex::mdspan<const Float, stdex::extents<int, d...>, stdex::layout_stride, stdex::default_accessor<const Float>>;	    
       using Map = stdex::layout_stride::mapping<stdex::extents<int, d...>>;
       //
       return View{ data_container[tid].data() + batch_id,
                    Map(stdex::extents<int, d...>{}, strides) };
    }

    template<ArithmeticTp dataTp, bool dummy = false>
    auto D1DynAccessor(const size_t idx, dyn_indx_type ext, std::array<dyn_indx_type, 1> strides) {
       using Map    = stdex::layout_stride::mapping<stdex::extents<dyn_indx_type, stdex::dynamic_extent>>;	 
       using View   = stdex::mdspan<dataTp, stdex::extents<dyn_indx_type, stdex::dynamic_extent>, stdex::layout_stride>;

       return View{ dummy == false ? data_container[idx].data() : nullptr, Map(stdex::extents<dyn_indx_type, stdex::dynamic_extent>{ext}, strides) };
    }  
};

template <typename vfnn_tp, typename vfn_tp, typename vf1_tp>
class KalmanArgs{
  public:
  
    vfnn_tp rqr;
    vfnn_tp t;    
    vfnn_tp p;       
    //     
    vfn_tp z;        
    vfn_tp alpha;
    //
    vf1_tp ys;            
    vf1_tp mu;
    vf1_tp vs;                    
    vf1_tp Fs;                    
    vf1_tp sum_logFs;                    
    //
    vf1_tp fc;            
    vf1_tp F_fc;
    //                    
    KalmanArgs(vfnn_tp &rqr, 
               vfnn_tp &t, 
               vfnn_tp &p, 
               //
               vfn_tp &z, 
               vfn_tp &alpha,
               //
               vf1_tp &ys,
               vf1_tp &mu,
               vf1_tp &vs,               
               vf1_tp &Fs,
               vf1_tp &sum_logFs,
               vf1_tp &fc,
               vf1_tp &F_fc) : rqr(rqr), t(t), p(p), z(z), alpha(alpha), ys(ys), mu(mu), vs(vs), Fs(Fs), sum_logFs(sum_logFs), fc(fc), F_fc(F_fc){ }
};

template <typename Arg, int d_, int bsize_>
class KalmanAlgo{
  public:
    static constexpr int d     = d_;
    static constexpr int bsize = bsize_;    
  
    const Arg &arg;
    
    int n_diff;
    
    const int nseries;
    const int nobs;
    
    KalmanAlgo(const Arg &arg, const int nseries, const int nobs) : arg(arg), n_diff(0), nseries(nseries), nobs(nobs) {} 
    
    void operator()(const auto idx) {
      //
      const auto tid      = idx / bSize;
      const auto batch_id = idx % bSize;                               
                              
      // First, create local objects and views:
      std::array<Float, d*d> rqr;
      const Left2DView rqr_{rqr.data()};
      //
      std::array<Float, d*d> t;//
      Left2DView    t_{t.data()};//
      //
      std::array<Float, d>  z;
      const Left1DView z_{z.data()};
      // 
      std::array<Float, d*d> p;
      Left2DView    p_{p.data()};//
      //
      std::array<Float, d>  a;
      Left1DView       a_{a.data()};//                              

      // Load global mem into registers 
      //      
      auto RQR_ = arg.rqr.template Accessor<d, d>(tid, batch_id, {bsize, bsize*d});                                                                                              
      //
      auto T_   = arg.t.template Accessor<d, d>(tid, batch_id, {bsize, bsize*d});
      //
      auto P_   = arg.p.template Accessor<d, d>(tid, batch_id, {bsize, bsize*d});
                
#pragma unroll    
      for (int i = 0; i < d; i++) {
#pragma unroll                              
        for (int j = 0; j < d; j++) {                                                             
          rqr_(i,j) = RQR_(i,j);
          t_(i,j)   = T_(i,j);
          p_(i,j)   = P_(i,j);
        }
      }
                              
      auto Z_    = arg.z.template Accessor<d>(tid, batch_id, {bsize});
      //
      auto alpha_= arg.alpha.template Accessor<d>(tid, batch_id, {bsize});
#pragma unroll    
      for (int i = 0; i < d; i++) {;                               
        if (n_diff > 0) z_(i) = Z_(i);
        a_(i)                 = alpha_(i);
      }

      auto ys_ = arg.ys.template D1DynAccessor<const Float>(static_cast<dyn_indx_type>(idx), static_cast<dyn_indx_type>(nobs), std::array<dyn_indx_type, 1>{nseries});
      //
      auto vs_ = arg.vs.template D1DynAccessor<Float>(static_cast<dyn_indx_type>(idx), static_cast<dyn_indx_type>(nobs), std::array<dyn_indx_type, 1>{nseries});

      auto Fs_ = arg.Fs.template D1DynAccessor<Float>(static_cast<dyn_indx_type>(idx), static_cast<dyn_indx_type>(nobs), std::array<dyn_indx_type, 1>{nseries});

      Float mu_ = intercept ? arg.mu.data_container[idx][0] : 0.0;//Float1
    
      Float sum_logFs_ = kalman_update<d>(a_, vs_, t_, p_, Fs_, ys_, rqr_, z_, mu_, n_diff, nseries, nobs);

      // Forecast
      constexpr bool is_empty = fc_steps ? false : true;  
      //                                            
      auto fc_ = arg.fc.template D1DynAccessor<Float, is_empty>(static_cast<dyn_indx_type>(idx), fc_steps, std::array<dyn_indx_type, 1>{nseries}); 
      auto F_fc_ = arg.F_fc.template D1DynAccessor<Float, is_empty>(static_cast<dyn_indx_type>(idx), fc_steps, std::array<dyn_indx_type, 1>{nseries});                                              

      kalman_forecast<d, forecast>(a_, fc_, t_, p_, F_fc_, rqr_, z_, mu_, n_diff, nseries, fc_steps);

      arg.sum_logFs.data_container[idx][0] = sum_logFs_;
    }
    
    auto GetNSeries() { return nseries; }
    void UpdateNDiff(const int n_diff_) { n_diff = n_diff_; }
};



int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: "<< argv[0] <<" <Series> <Repeat>\n" << std::endl;
    exit(-2);
  }  
  
  const size_t nseries   = atoi(argv[1]); 
  
  if (nseries < bSize) {
    std::cout << "Number of series is smaller then batch size ( " << bSize << " )\n" << std::endl;
    exit(-1);
  }   
  
  const int repeat    = atoi(argv[2]);
  
  const size_t bnseries = nseries / bSize;  
  
  std::default_random_engine gen;
  std::uniform_real_distribution<Float> dist(0.0,1.0);

  auto RQR = GenericContainerWrapper<vFloatNN>(bnseries, gen, dist);
  auto tRQR = RQR.Get();  

  auto T   = GenericContainerWrapper<vFloatNN>(bnseries, gen, dist);
  auto&& tT = T.Get();    

  auto P   = GenericContainerWrapper<vFloatNN>(bnseries, gen, dist);
  auto&& tP = P.Get();  
  
  auto Z   = GenericContainerWrapper<vFloatN>(bnseries, gen, dist);
  auto&& tZ   = Z.Get();       
  
  auto alpha = GenericContainerWrapper<vFloatN>(bnseries, gen, dist);
  auto&& talpha   = alpha.Get();         
  
  auto ys = GenericContainerWrapper<vFloat1>(nseries*nObs, gen, dist);
  auto&& tys  = ys.Get();  
  
  auto mu = GenericContainerWrapper<vFloat1>(nseries, gen, dist);
  auto&& tmu  = mu.Get();    

  auto vs = GenericContainerWrapper<vFloat1>(nseries*nObs);
  auto&& tvs  = vs.Get();      

  auto Fs = GenericContainerWrapper<vFloat1>(nseries*nObs);
  auto&& tFs  = Fs.Get();        
  //
  auto sum_logFs = GenericContainerWrapper<vFloat1>(nseries);
  auto&& tsum_logFs  = sum_logFs.Get();          
  //
  auto fc   = GenericContainerWrapper<vFloat1>(nseries * fc_steps);
  auto&& tfc  = fc.Get();            
  //
  auto F_fc = GenericContainerWrapper<vFloat1>(nseries * fc_steps);
  auto&& tF_fc  = F_fc.Get();  

  /**
  * Kalman loop kernel. Each thread computes kalman filter for a single series
  * and stores relevant matrices in registers.
  */  
  using vFNNtp = GenericContainerWrapper<std::span<typename vFloatNN::value_type>>;
  using vFNtp  = GenericContainerWrapper<std::span<typename vFloatN::value_type>>;  
  using vF1tp  = GenericContainerWrapper<std::span<typename vFloat1::value_type>>;
  
  std::unique_ptr<KalmanArgs<vFNNtp, vFNtp, vF1tp>> kalman_args_ptr(new KalmanArgs{tRQR, tT, tP, tZ, talpha, tys, tmu, tvs, tFs, tsum_logFs, tfc, tF_fc});
  
  auto& kalman_args = *kalman_args_ptr;
  
  auto kalman_algo   = std::make_shared<KalmanAlgo<decltype(kalman_args), D, bSize>>(kalman_args, nseries, nObs);

  auto kalman_kernel = [&kalman_algo = *kalman_algo.get()] (const auto i) { kalman_algo(i); };  

  for (int n_diff = 0; n_diff < D; n_diff++) {
    
    auto start = std::chrono::steady_clock::now();
    //
    for (int i = 0; i < repeat; i++) {
      dispatch_kalman_kernel(kalman_kernel, kalman_algo->GetNSeries());
    }
    
    kalman_algo->UpdateNDiff(n_diff);
    
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time (n_diff = " << n_diff << " ): " << (time * 1e-9f) / repeat <<" (s)\n" << std::endl;
  }

  double sum = 0.0;
  
  for (int i = 0; i < fc_steps * nseries - 1; i++)
    sum += fabs((fabs(F_fc.data_container[i+1][0]) - fabs(F_fc.data_container[i][0]))) / (fabs(F_fc.data_container[i+1][0]) + fabs(F_fc.data_container[i][0]));
    
  std::cout << "Checksum: " << sum << std::endl;
  
  return 0;
}
