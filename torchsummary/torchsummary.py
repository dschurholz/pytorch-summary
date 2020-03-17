import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np


conv_modules = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.Conv1d, nn.Conv2d,
                nn.Conv3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)
linear_modules = (nn.Linear, )
dropout_modules = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
batch_norm_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def expand_kernel(k, layer_class):
    if type(k) is list:
        return k
    if type(k) is tuple:
        return list(k)
    num_dim = 3 if '3' in layer_class else (2 if '2' in layer_class else 1)
    return [k] * num_dim


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'),
            dtypes=None):
    result, params = summary_string(
        model, input_size, batch_size, device, dtypes)
    return result


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'),
                   dtypes=None):
    if dtypes is None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            if module.__class__ in conv_modules:
                summary[m_key]["w_shape"] = "k: {0}, p: {1}, s: {2}".format(
                    str(expand_kernel(
                        module.kernel_size, module.__class__.__name__)),
                    str(expand_kernel(
                        module.padding, module.__class__.__name__)),
                    str(expand_kernel(
                        module.stride, module.__class__.__name__))
                )
            elif module.__class__ in linear_modules:
                summary[m_key]["w_shape"] = (
                    "# in features: {0}, # out features: {1}".format(
                        str(module.in_features), str(module.out_features)))
            elif module.__class__ in dropout_modules:
                summary[m_key]["w_shape"] = "percentage: {0:.0f}%".format(
                    module.p * 100)
            elif module.__class__ in batch_norm_modules:
                summary[m_key]["w_shape"] = "# features: {0}".format(
                    str(module.num_features))
            else:
                summary[m_key]["w_shape"] = ''
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    template = "{:>20}  {:>45}  {:>25} {:>15}"
    summary_str += ("-" * 110) + "\n"
    line_new = template.format(
        "Layer (type)", "Layer Shape", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += ("=" * 110) + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    summary_str += template.format(
        'Input', '', str(summary[list(summary.keys())[0]]["input_shape"]), '')
    summary_str += '\n'
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = template.format(
            layer,
            str(summary[layer]["w_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] is True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += ("=" * 110) + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += ("-" * 110) + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += ("-" * 110) + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)
