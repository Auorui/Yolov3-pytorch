cfg = {
        "yes":               True,
        "no":                False,
        "classes_path":      'model_data/classes.txt',
        "voc_path":          r'VOCdevkit',
        "anchors_path":      'model_data/yolo_anchors.txt',
        "anchors_mask":      [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "input_shape":       [416, 416],
        "model_path_pth":    r'model_data/yolo_weights.pth',
        "no_mode_path":      '',
        "save_dir":          r'logs',
        "save_period":       20,
        "optimizer_type":    "sgd",                   # "sgd"  and  "adam"
        "lr_decay_type":     "cos",                   # "cos"  and  "step"
        "eval_period":        1,
}

help = {
        "yes":               "bool值,表示为 True ",
        "no":                "bool值,表示为 False ",
        "classes_path":      "指向model_data下的txt,与自己训练的数据集相关,使用自己的数据集一定要修改classes.txt文件内容",
        "voc_path":          "指向voc数据集的目录,大致结构无需修改",
        "anchors_path":      "代表先验框对应的txt文件,一般不修改",
        "anchors_mask":      "用于帮助代码找到对应的先验框,一般不修改",
        "input_shape":       "输入的shape大小，一定要是32的倍数",
        "model_path_pth":    "模型的预训练权重,如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训\
                             练了一部分的权值再次载入。同时修改下方的冻结阶段或者解冻阶段的参数，来保证模型epoch的连续性。",
        "no_mode_path":      "不加载整个模型的权值。",
        "save_dir":          "权值与日志文件保存的文件夹",
        "save_period":       "每隔多少epoch保存一次权值",
        "optimizer_type":    "使用到的优化器种类，可选的有adam、sgd",
        "lr_decay_type":     "使用到的学习率下降方式，可选的有step、cos",
        "eval_period":       "代表多少个epoch评估一次，不建议频繁的评估,评估需要消耗较多的时间，频繁评估会导致训练非常慢",
}
