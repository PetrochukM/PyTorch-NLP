extern "C" {
  __forceinline__ __device__ float sigmoidf(float x)
  {
      return 1.f / (1.f + expf(-x));
  }
  __forceinline__ __device__ float reluf(float x)
  {
      return (x > 0.f) ? x : 0.f;
  }
  __forceinline__ __device__ float seluf(float x)
  {
      return 1.0507009873554804934193349852946f * (
          (x > 0.f) ? x : 1.6732632423543772848170429916717f * (expf(x)-1.f)
      );
  }
  __forceinline__ __device__ float calc_activation(int type, float x)
  {
      switch (type) {
          case 0:
              return x;
          case 1:
              return tanh(x);
          case 2:
              return reluf(x);
          case 3:
              return seluf(x);
      }
      return x;
  }
  __forceinline__ __device__ float calc_grad_activation(int type, float x)
  {
      switch (type) {
          case 0:
              return 1.f;
          case 1:
              return 1.f-x*x;
          case 2:
              return (x > 0.f) ? 1.f : 0.f;
          case 3:
              return (x > 0.f) ? 1.0507009873554804934193349852946f :
                  x + 1.7580993408473766f;
      }
      return 1.f;
  }
  __global__ void sru_fwd(const float * __restrict__ u, const float * __restrict__ x,
                          const float * __restrict__ bias, const float * __restrict__ init,
                          const float * __restrict__ mask_h,
                          const int len, const int batch, const int d, const int k,
                          float * __restrict__ h, float * __restrict__ c,
                          const int activation_type)
  {
      assert ((k == 3) || (x == NULL));
      int ncols = batch*d;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= ncols) return;
      int ncols_u = ncols*k;
      int ncols_x = (k == 3) ? ncols : ncols_u;
      const float bias1 = *(bias + (col%d));
      const float bias2 = *(bias + (col%d) + d);
      const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
      float cur = *(init + col);
      const float *up = u + (col*k);
      const float *xp = (k == 3) ? (x + col) : (up + 3);
      float *cp = c + col;
      float *hp = h + col;
      for (int row = 0; row < len; ++row)
      {
          float g1 = sigmoidf((*(up+1))+bias1);
          float g2 = sigmoidf((*(up+2))+bias2);
          cur = (cur-(*up))*g1 + (*up);
          *cp = cur;
          float val = calc_activation(activation_type, cur);
          *hp = (val*mask-(*xp))*g2 + (*xp);
          up += ncols_u;
          xp += ncols_x;
          cp += ncols;
          hp += ncols;
      }
  }
  __global__ void sru_bwd(const float * __restrict__ u, const float * __restrict__ x,
                          const float * __restrict__ bias, const float * __restrict__ init,
                          const float * __restrict__ mask_h, const float * __restrict__ c,
                          const float * __restrict__ grad_h, const float * __restrict__ grad_last,
                          const int len, const int batch, const int d, const int k,
                          float * __restrict__ grad_u, float * __restrict__ grad_x,
                          float * __restrict__ grad_bias, float * __restrict__ grad_init,
                          int activation_type)
  {
      assert((k == 3) || (x == NULL));
      assert((k == 3) || (grad_x == NULL));
      int ncols = batch*d;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= ncols) return;
      int ncols_u = ncols*k;
      int ncols_x = (k == 3) ? ncols : ncols_u;
      const float bias1 = *(bias + (col%d));
      const float bias2 = *(bias + (col%d) + d);
      const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
      float gbias1 = 0;
      float gbias2 = 0;
      float cur = *(grad_last + col);
      const float *up = u + (col*k) + (len-1)*ncols_u;
      const float *xp = (k == 3) ? (x + col + (len-1)*ncols) : (up + 3);
      const float *cp = c + col + (len-1)*ncols;
      const float *ghp = grad_h + col + (len-1)*ncols;
      float *gup = grad_u + (col*k) + (len-1)*ncols_u;
      float *gxp = (k == 3) ? (grad_x + col + (len-1)*ncols) : (gup + 3);
      for (int row = len-1; row >= 0; --row)
      {
          const float g1 = sigmoidf((*(up+1))+bias1);
          const float g2 = sigmoidf((*(up+2))+bias2);
          const float c_val = calc_activation(activation_type, *cp);
          const float x_val = *xp;
          const float u_val = *up;
          const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));
          const float gh_val = *ghp;
          // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
          // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0
          // grad wrt x
          *gxp = gh_val*(1-g2);
          // grad wrt g2, u2 and bias2
          float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
          *(gup+2) = gg2;
          gbias2 += gg2;
          // grad wrt c
          const float tmp = g2*calc_grad_activation(activation_type, c_val);
          const float gc = gh_val*mask*tmp + cur;
          // grad wrt u0
          *gup = gc*(1-g1);
          // grad wrt g1, u1, and bias1
          float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
          *(gup+1) = gg1;
          gbias1 += gg1;
          // grad wrt c'
          cur = gc*g1;
          up -= ncols_u;
          xp -= ncols_x;
          cp -= ncols;
          gup -= ncols_u;
          gxp -= ncols_x;
          ghp -= ncols;
      }
      *(grad_bias + col) = gbias1;
      *(grad_bias + col + ncols) = gbias2;
      *(grad_init +col) = cur;
  }
  __global__ void sru_bi_fwd(const float * __restrict__ u, const float * __restrict__ x,
                          const float * __restrict__ bias, const float * __restrict__ init,
                          const float * __restrict__ mask_h,
                          const int len, const int batch, const int d, const int k,
                          float * __restrict__ h, float * __restrict__ c,
                          const int activation_type)
  {
      assert ((k == 3) || (x == NULL));
      assert ((k == 3) || (k == 4));
      int ncols = batch*d*2;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= ncols) return;
      int ncols_u = ncols*k;
      int ncols_x = (k == 3) ? ncols : ncols_u;
      const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
      float cur = *(init + col);
      const int d2 = d*2;
      const bool flip = (col%d2) >= d;
      const float bias1 = *(bias + (col%d2));
      const float bias2 = *(bias + (col%d2) + d2);
      const float *up = u + (col*k);
      const float *xp = (k == 3) ? (x + col) : (up + 3);
      float *cp = c + col;
      float *hp = h + col;
      if (flip) {
          up += (len-1)*ncols_u;
          xp += (len-1)*ncols_x;
          cp += (len-1)*ncols;
          hp += (len-1)*ncols;
      }
      int ncols_u_ = flip ? -ncols_u : ncols_u;
      int ncols_x_ = flip ? -ncols_x : ncols_x;
      int ncols_ = flip ? -ncols : ncols;
      for (int cnt = 0; cnt < len; ++cnt)
      {
          float g1 = sigmoidf((*(up+1))+bias1);
          float g2 = sigmoidf((*(up+2))+bias2);
          cur = (cur-(*up))*g1 + (*up);
          *cp = cur;
          float val = calc_activation(activation_type, cur);
          *hp = (val*mask-(*xp))*g2 + (*xp);
          up += ncols_u_;
          xp += ncols_x_;
          cp += ncols_;
          hp += ncols_;
      }
  }
  __global__ void sru_bi_bwd(const float * __restrict__ u, const float * __restrict__ x,
                          const float * __restrict__ bias, const float * __restrict__ init,
                          const float * __restrict__ mask_h, const float * __restrict__ c,
                          const float * __restrict__ grad_h, const float * __restrict__ grad_last,
                          const int len, const int batch, const int d, const int k,
                          float * __restrict__ grad_u, float * __restrict__ grad_x,
                          float * __restrict__ grad_bias, float * __restrict__ grad_init,
                          int activation_type)
  {
      assert((k == 3) || (x == NULL));
      assert((k == 3) || (grad_x == NULL));
      assert((k == 3) || (k == 4));
      int ncols = batch*d*2;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= ncols) return;
      int ncols_u = ncols*k;
      int ncols_x = (k == 3) ? ncols : ncols_u;
      const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
      float gbias1 = 0;
      float gbias2 = 0;
      float cur = *(grad_last + col);
      const int d2 = d*2;
      const bool flip = ((col%d2) >= d);
      const float bias1 = *(bias + (col%d2));
      const float bias2 = *(bias + (col%d2) + d2);
      const float *up = u + (col*k);
      const float *xp = (k == 3) ? (x + col) : (up + 3);
      const float *cp = c + col;
      const float *ghp = grad_h + col;
      float *gup = grad_u + (col*k);
      float *gxp = (k == 3) ? (grad_x + col) : (gup + 3);
      if (!flip) {
          up += (len-1)*ncols_u;
          xp += (len-1)*ncols_x;
          cp += (len-1)*ncols;
          ghp += (len-1)*ncols;
          gup += (len-1)*ncols_u;
          gxp += (len-1)*ncols_x;
      }
      int ncols_u_ = flip ? -ncols_u : ncols_u;
      int ncols_x_ = flip ? -ncols_x : ncols_x;
      int ncols_ = flip ? -ncols : ncols;
      for (int cnt = 0; cnt < len; ++cnt)
      {
          const float g1 = sigmoidf((*(up+1))+bias1);
          const float g2 = sigmoidf((*(up+2))+bias2);
          const float c_val = calc_activation(activation_type, *cp);
          const float x_val = *xp;
          const float u_val = *up;
          const float prev_c_val = (cnt<len-1) ? (*(cp-ncols_)) : (*(init+col));
          const float gh_val = *ghp;
          // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
          // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0
          // grad wrt x
          *gxp = gh_val*(1-g2);
          // grad wrt g2, u2 and bias2
          float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
          *(gup+2) = gg2;
          gbias2 += gg2;
          // grad wrt c
          const float tmp = g2*calc_grad_activation(activation_type, c_val);
          const float gc = gh_val*mask*tmp + cur;
          // grad wrt u0
          *gup = gc*(1-g1);
          // grad wrt g1, u1, and bias1
          float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
          *(gup+1) = gg1;
          gbias1 += gg1;
          // grad wrt c'
          cur = gc*g1;
          up -= ncols_u_;
          xp -= ncols_x_;
          cp -= ncols_;
          gup -= ncols_u_;
          gxp -= ncols_x_;
          ghp -= ncols_;
      }
      *(grad_bias + col) = gbias1;
      *(grad_bias + col + ncols) = gbias2;
      *(grad_init +col) = cur;
  }
}