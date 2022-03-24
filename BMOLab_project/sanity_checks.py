
import torch

# rn101_trainable_layers = ["conv1", "conv2", "conv3", "layer1:0", "layer1:1",
#                           "layer1:2", "layer2:0", "layer2:1", "layer2:2", "layer2:3", "layer3:0", "layer3:1",
#                           "layer3:2", "layer3:3", "layer3:4", "layer3:5", "layer3:6", "layer3:7", "layer3:8",
#                           "layer3:9", "layer3:10", "layer3:11", "layer3:12", "layer3:13", "layer3:14", "layer3:15",
#                           "layer3:16", "layer3:17", "layer3:18", "layer3:19", "layer3:20", "layer3:21", "layer3:22",
#                           "layer4:0", "layer4:1", "layer4:2", "attnpool"]
#
# vit_trainable_layers = ["conv1", "ln_pre", "transformer:0", "transformer:1", "transformer:2", "transformer:3",
#                         "transformer:4", "transformer:5", "transformer:6", "transformer:7", "transformer:8",
#                         "transformer:9", "transformer:10", "transformer:11", "ln_post"]


def reset_model_layers(model, model_name, layers):
    if model_name == "RN101":
        for name, layer in list(model.visual.named_children()):
            if name in layers and hasattr(layer, "reset_parameters"):
                print(f'Reset trainable parameters (in {model_name}) of layer = {name}\n')
                layer.reset_parameters()

            if name.startswith("layer"):
                for subname, sublayer in list(layer.named_children()):
                    full_subname = name + ":" + subname

                    if full_subname in layers:  # reset bottleneck layer
                        print(f'Reset trainable parameters (in {model_name}) of sublayer = {full_subname}')
                        reset_block_layer(sublayer, full_subname, model_name)

            if name == "attnpool" and name in layers:
                print(f'Reset trainable parameters (in {model_name}) of sublayer = {name}')
                reset_block_layer(layer, name, model_name)

    if model_name == "ViT":
        for name, layer in list(model.visual.named_children()):
            if name in layers and hasattr(layer, "reset_parameters"):
                print(f'Reset trainable parameters (in {model_name}) of layer = {name}\n')
                layer.reset_parameters()

            if name == "transformer":
                resblocks, resblocks_layers = list(layer.named_children())[0]

                for subname, sublayer in list(resblocks_layers.named_children()):
                    full_subname = name + ":" + subname

                    if full_subname in layers:  # reset residualattentionblock layer
                        print(f'Reset trainable parameters (in {model_name}) of sublayer = {full_subname}')
                        reset_block_layer(sublayer, full_subname, model_name)


def reset_block_layer(block, block_name, model_name):
    for name, layer in list(block.named_children()):
        if hasattr(layer, "reset_parameters"):
            print(f'Reset trainable parameters (in {model_name} block {block_name}) of sublayer = {name}')
            layer.reset_parameters()

        if name == "downsample":
            for subname, sublayer in list(layer.named_children()):
                if hasattr(sublayer, "reset_parameters"):
                    print(f'Reset trainable parameters (in {model_name} block {block_name}) of sublayer = downsample:{subname}')
                    sublayer.reset_parameters()

        if name == "attn":
            for subname, sublayer in list(layer.named_children()):
                if hasattr(sublayer, "reset_parameters"):
                    print(f'Reset trainable parameters (in {model_name} block {block_name}) of sublayer = attn:{subname}')
                    sublayer.reset_parameters()

        if name == "mlp":
            for subname, sublayer in list(layer.named_children()):
                if hasattr(sublayer, "reset_parameters"):
                    print(f'Reset trainable parameters (in {model_name} block {block_name}) of sublayer = mlp:{subname}')
                    sublayer.reset_parameters()
    print("")