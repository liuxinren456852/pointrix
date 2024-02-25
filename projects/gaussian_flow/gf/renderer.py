import torch
from pointrix.renderer.base_splatting import GaussianSplattingRender, RENDERER_REGISTRY

@RENDERER_REGISTRY.register()
class GaussianFlowRenderer(GaussianSplattingRender):

    def render_batch(self, model, batch):
        """
        Render a batch of views.

        Parameters
        ----------
        render_dict : dict
            The rendering dictionary.
        batch : list
            The batch of views.
        """
        renders = []
        viewspace_points = []
        visibilitys = []
        radiis = []
        
        # get the static render dict once and use it for all the batch
        static_render_dict = model.get_gaussian()
        
        def render_func(data):
            data.update(static_render_dict)
            # set timestep for each data in the batch
            dynamic_render_dict = model(data)
            data.update(dynamic_render_dict)
            return self.render_iter(**data)
        
        for b_i in batch:
            render_results = render_func(b_i)
            renders.append(render_results["render"])
            viewspace_points.append(render_results["viewspace_points"])
            visibilitys.append(
                render_results["visibility_filter"].unsqueeze(0)
            )
            radiis.append(render_results["radii"].unsqueeze(0))

        radii = torch.cat(radiis, 0).max(dim=0).values
        visibility = torch.cat(visibilitys).any(dim=0)
        images = torch.stack(renders)

        render_results = {
            "images": images,
            "radii": radii,
            "visibility": visibility,
            "viewspace_points": viewspace_points,
        }

        return render_results