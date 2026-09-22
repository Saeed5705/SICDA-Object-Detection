import torch



class SICDACollate:


    def __call__(
        self,
        batch
    ):


        rgb_images = []
        thermal_images = []

        targets = []

        rgb_paths = []
        thermal_paths = []



        for item in batch:


            rgb_images.append(
                item["rgb"]
            )


            thermal_images.append(
                item["thermal"]
            )



            # Source / Target annotations

            if "target" in item:

                targets.append(
                    item["target"]
                )


            else:

                targets.append(
                    None
                )



            rgb_paths.append(
                item.get(
                    "rgb_path",
                    None
                )
            )


            thermal_paths.append(
                item.get(
                    "thermal_path",
                    None
                )
            )



        # Stack images

        rgb_images = torch.stack(
            rgb_images,
            dim=0
        )


        thermal_images = torch.stack(
            thermal_images,
            dim=0
        )



        return {


            "rgb": rgb_images,


            "thermal": thermal_images,


            "targets": targets,


            "rgb_path": rgb_paths,


            "thermal_path": thermal_paths

        }