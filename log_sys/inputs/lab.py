from pprint import pprint 

pprint("""[34m{[0m
    [32membeddings[0m[35m:[0m [34m{[0m
        [32mLayerNorm[0m[35m:[0m [34m{[0m
            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
        [34m}[0m,
        [32mposition_embeddings[0m[35m:[0m [34m{[0m
            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[512, 768])
        [34m}[0m,
        [32mtoken_type_embeddings[0m[35m:[0m [34m{[0m
            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[2, 768])
        [34m}[0m,
        [32mword_embeddings[0m[35m:[0m [34m{[0m
            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[30522, 768])
        [34m}[0m
    [34m}[0m,
    [32mencoder[0m[35m:[0m [34m{[0m
        [32mlayer[0m[35m:[0m [34m{[0m
            [32mv0[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv1[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv10[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv11[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv2[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv3[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv4[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv5[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv6[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv7[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv8[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32mv9[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mffd[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense1[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m,
                    [32mdense2[0m[35m:[0m [34m{[0m
                        [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m
        [34m}[0m
    [34m}[0m,
    [32mpooler[0m[35m:[0m [34m{[0m
        [32mdense[0m[35m:[0m [34m{[0m
            [32mb[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
            [32mw[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
        [34m}[0m
    [34m}[0m
[34m}[0m
[34m{[0m
    [32membeddings[0m[35m:[0m [34m{[0m
        [32mLayerNorm[0m[35m:[0m [34m{[0m
            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
        [34m}[0m,
        [32mposition_embeddings[0m[35m:[0m [34m{[0m
            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[512, 768])
        [34m}[0m,
        [32mtoken_type_embeddings[0m[35m:[0m [34m{[0m
            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[2, 768])
        [34m}[0m,
        [32mword_embeddings[0m[35m:[0m [34m{[0m
            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[30522, 768])
        [34m}[0m
    [34m}[0m,
    [32mencoder[0m[35m:[0m [34m{[0m
        [32mlayer[0m[35m:[0m [34m{[0m
            [32m0[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m1[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m10[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m11[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m2[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m3[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m4[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m5[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m6[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m7[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m8[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m,
            [32m9[0m[35m:[0m [34m{[0m
                [32mattention[0m[35m:[0m [34m{[0m
                    [32moutput[0m[35m:[0m [34m{[0m
                        [32mLayerNorm[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                        [34m}[0m,
                        [32mdense[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m,
                    [32mself[0m[35m:[0m [34m{[0m
                        [32mkey[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mquery[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m,
                        [32mvalue[0m[35m:[0m [34m{[0m
                            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
                        [34m}[0m
                    [34m}[0m
                [34m}[0m,
                [32mintermediate[0m[35m:[0m [34m{[0m
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[3072, 768])
                    [34m}[0m
                [34m}[0m,
                [32moutput[0m[35m:[0m [34m{[0m
                    [32mLayerNorm[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768])
                    [34m}[0m,
                    [32mdense[0m[35m:[0m [34m{[0m
                        [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
                        [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 3072])
                    [34m}[0m
                [34m}[0m
            [34m}[0m
        [34m}[0m
    [34m}[0m,
    [32mpooler[0m[35m:[0m [34m{[0m
        [32mdense[0m[35m:[0m [34m{[0m
            [32mbias[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768]),
            [32mweight[0m[35m:[0m (<[34mclass[0m ivy.data_classes.array.array.Array> [35mshape=[0m[768, 768])
        [34m}[0m
    [34m}[0m
[34m}[0m
""")