from typing import List, Optional, Sequence, Union
import numpy as np
from sklearn import metrics
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from typing import Dict
import torch
class ClassifyEval(object):
    def __init__(self,classDct,cocoGt):
        super().__init__()
        self.classDct=dict(classDct)
        self.classlst=[kk for kk in self.classDct.keys()]
        self.classlst.append('NoneClass')
        self.oriClass=self.dataset_meta['classes']
        assert(len(self.classDct.keys()) !=0)
        self.gtdct={}
        for img in cocoGt.imgs.values():
            id=img['id']
            curClass=None
            if id not in cocoGt.imgToAnns:
                curClass='NoneClass'
            else:
                curClass=self.GetCurClass(cocoGt,id)
            self.gtdct[id]=self.classlst.index(curClass)
        return
    def GetCurClass(self,cocoGt,id):
        img2Ann=cocoGt.imgToAnns[id]
        lst=[]
        for m in img2Ann:
            lst.append(self.oriClass[m['category_id']])
        res=set(lst)
        curKey='NoneClass'
        for kk,vv in self.classDct.items():
            if('include' in vv):
                bFlag=True
                for v in vv['include']:
                    if v not in res:bFlag=False
                if(bFlag):
                    return kk
                continue
            if('oneof' in vv):
                for v in vv['oneof']:
                    if v in res:
                        return kk
                continue
            if('exclude' in vv):
                bFlag=False
                for v in vv['exclude']:
                    if v in res:
                        bFlag=True
                        break
                if(not bFlag):
                    return kk
                continue
        return curKey
    def evaluate(self,results):
        preds=[]
        targets=[]
        predlabels=[]
        for i,res in enumerate(results):
            arr=res[1]['tailcls']
            img_id=res[0]['img_id']
            predlabel=np.array(arr,np.float32)
            predlabel=np.argmax(predlabel)
            predlabels.append(predlabel)
            preds.append(arr)
            targets.append(self.gtdct[img_id])
        labeldex=np.eye(len(self.classlst))
        labeltarget=[]
        resdct={}
        for i in range(len(targets)):
            labeltarget.append(labeldex[targets[i]].tolist())       
        try: 
            val=metrics.roc_auc_score(labeltarget,preds,multi_class='ovo')
            prec=metrics.precision_score(targets,predlabels,average=None)
            recall=metrics.recall_score(targets,predlabels,average=None)
            f1score=metrics.f1_score(targets,predlabels,average=None)
            for i in range(len(prec)):
                resdct['prec'+str(i)]=prec[i]
                resdct['recall'+str(i)]=recall[i]
                resdct['f1score'+str(i)]=f1score[i]
            resdct['auc']=val
        except Exception as e:
            print(e.args)
        return resdct
'''
test_evaluator = dict(
    type='CocoMetricWithCls',
    classDct=dct,
    ann_file=workDir+'test/cocofile.json',
    metric='bbox',
    format_only=False)
'''
@METRICS.register_module()
class CocoMetricWithCls(CocoMetric,ClassifyEval):
    def __init__(self, classDct:dict,*args,**kwargs) -> None:
        CocoMetric.__init__(self,args,kwargs)
        ClassifyEval.__init__(self,classDct,self._coco_api)
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['tailcls']=pred['tailcls'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))
    def compute_metrics(self, results: list) -> Dict[str, float]:
        eval_results=CocoMetric.compute_metrics(self,results)
        eval_results.update(ClassifyEval.evaluate(self,results))
        return eval_results

