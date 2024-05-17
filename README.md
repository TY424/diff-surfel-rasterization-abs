# Differential Surfel Rasterization-Abs

This rasterization engine incorporates AbsGS into 2D-GS. 

Result of dataset `360v2 bicycle` : 

<p align="center">
  <a href="">
    <img src="./README.assets/image-20240517213507725.png" width="99%">
  </a>
</p>

> left: 2D-GS 
>
> right: 2D-GS with AbsGS 



## Usage

```python


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
  
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.get_xyz.shape[0],4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
	...

def add_densification_stats(self, viewspace_point_tensor, update_filter):
    # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
    self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                         keepdim=True)
    self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1,
                                                             keepdim=True)
    self.denom[update_filter] += 1    
```





## BibTeX

```
@misc{ye2024absgs,
      title={AbsGS: Recovering Fine Details for 3D Gaussian Splatting}, 
      author={Zongxin Ye and Wenyu Li and Sidun Liu and Peng Qiao and Yong Dou},
      year={2024},
      eprint={2404.10484},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}

@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

