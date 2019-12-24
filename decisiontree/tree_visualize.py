from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt


def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        num_leafs += get_num_leafs(second_dict[key]) if isinstance(second_dict[key], dict) else 1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        depth = 1 + get_tree_depth(second_dict[key]) if isinstance(second_dict[key], dict) else 1
        if depth > max_depth:
            max_depth = depth
    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
    create_plot.ax1.annotate(node_txt, xy=parent_pt,  xycoords='axes fraction',
                            xytext=center_pt, textcoords='axes fraction',
                            va="center", ha="center", bbox=node_type, arrowprops=arrow_args, FontProperties=font)


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0]-cntr_pt[0])/2.0 + cntr_pt[0]
    y_mid = (parent_pt[1]-cntr_pt[1])/2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(my_tree, parent_pt, node_txt):
    decision_node = dict(boxstyle="sawtooth", fc="0.8")
    leaf_node = dict(boxstyle="round4", fc="0.8")
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = next(iter(my_tree))
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs))/2.0/plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            plot_tree(second_dict[key],cntr_pt,str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')   # 创建fig
    fig.clf()   # 清空fig
    ax_props = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **ax_props)   # 去掉x、y轴
    plot_tree.totalW = float(get_num_leafs(in_tree))   # 获取决策树叶结点数目
    plot_tree.totalD = float(get_tree_depth(in_tree))  # 获取决策树层数
    plot_tree.xOff = -0.5/plot_tree.totalW
    plot_tree.yOff = 1.0   # x 偏移
    plot_tree(in_tree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()
