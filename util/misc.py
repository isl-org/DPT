import matplotlib.pyplot as plt

from dpt.vit import get_mean_attention_map

def visualize_attention(input, model, prediction, model_type):
    input = (input + 1.0)/2.0

    attn1 = model.pretrained.attention["attn_1"]
    attn2 = model.pretrained.attention["attn_2"]
    attn3 = model.pretrained.attention["attn_3"]
    attn4 = model.pretrained.attention["attn_4"]

    plt.subplot(3,4,1), plt.imshow(input.squeeze().permute(1,2,0)), plt.title("Input", fontsize=8), plt.axis("off")
    plt.subplot(3,4,2), plt.imshow(prediction), plt.set_cmap("inferno"), plt.title("Prediction", fontsize=8), plt.axis("off")

    if model_type == "dpt_hybrid":
        h = [3,6,9,12]
    else:
        h = [6,12,18,24]

    # upper left
    plt.subplot(345),
    ax1 = plt.imshow(get_mean_attention_map(attn1, 1, input.shape))
    plt.ylabel("Upper left corner", fontsize=8)
    plt.title(f"Layer {h[0]}", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])


    plt.subplot(346),
    plt.imshow(get_mean_attention_map(attn2, 1, input.shape))
    plt.title(f"Layer {h[1]}", fontsize=8)
    plt.axis("off"),

    plt.subplot(347),
    plt.imshow(get_mean_attention_map(attn3, 1, input.shape))
    plt.title(f"Layer {h[2]}", fontsize=8)
    plt.axis("off"),


    plt.subplot(348),
    plt.imshow(get_mean_attention_map(attn4, 1, input.shape))
    plt.title(f"Layer {h[3]}", fontsize=8)
    plt.axis("off"),


    # lower right
    plt.subplot(3,4,9), plt.imshow(get_mean_attention_map(attn1, -1, input.shape))
    plt.ylabel("Lower right corner", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])

    plt.subplot(3,4,10), plt.imshow(get_mean_attention_map(attn2, -1, input.shape)), plt.axis("off")
    plt.subplot(3,4,11), plt.imshow(get_mean_attention_map(attn3, -1, input.shape)), plt.axis("off")
    plt.subplot(3,4,12), plt.imshow(get_mean_attention_map(attn4, -1, input.shape)), plt.axis("off")
    plt.tight_layout()
    plt.show()
