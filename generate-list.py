import tvm
from tvm import relay
from tvm.relay import analysis_tools
import pandas as pd
from tvm.relay.testing import mlp
from tvm.relay.testing import resnet
from tvm.relay.testing import dqn
from tvm.relay.testing import dcgan
from tvm.relay.testing import mobilenet
from tvm.relay.testing import lstm
from tvm.relay.testing import inception_v3
from tvm.relay.testing import squeezenet
from tvm.relay.testing import vgg
from tvm.relay.testing import densenet


class GetReadableName(analysis_tools.AnalysisPass):
    def visit_call(self, call):
        super().visit_call(call)
        self._add_detail(call, readable_name=call.op.name)


class GetIndex(analysis_tools.AnalysisPass):
    def __init__(self):
        super().__init__()
        self.__id = 0

    def visit_call(self, call):
        super().visit_call(call)
        self._add_detail(call, id=self.__id)
        self.__id += 1


class SummarizeOpTypes(relay.analysis_tools.AnalysisPass):
    def _summarize(self):
        histogram = {}
        for node, data in self._existing_data.items():
            if data['readable_name'] not in histogram:
                histogram[data['readable_name']] = 1
            else:
                histogram[data['readable_name']] += 1
        self._add_summary(histogram)


summaries = {}
summary_columns = set()
for (module, _), name in [
    (resnet.get_workload(num_layers=18), 'resnet18'),
    (resnet.get_workload(num_layers=50), 'resnet50'),
    (mobilenet.get_workload(), 'mobilenet'),
    (mlp.get_workload(batch_size=1), 'mlp'),
    (dqn.get_workload(batch_size=1), 'dqn'),
    (dcgan.get_workload(batch_size=1), 'dcgan'),
        # LSTM throws an error w/ analysis framework
        #    (lstm.get_workload(iterations=32, num_hidden=32), 'lstm'),
    (inception_v3.get_workload(), 'inception_v3'),
    (squeezenet.get_workload(), 'squeezenet'),
    (vgg.get_workload(batch_size=1), 'vgg'),
    (densenet.get_workload(), 'densenet'),
]:
    program = module['main']
    analyses = [GetReadableName(), GetIndex(), SummarizeOpTypes()]
    _, summary_results = relay.analysis_tools.run_analyses(program, analyses)
    summary_columns.update(
        relay.analysis_tools.get_summary_columns(summary_results))
    summaries[name] = summary_results

summary_columns_ordered = (sorted(list(summary_columns)))
summary_column_names = list(map(lambda t: t[0], summary_columns_ordered))
summary_records = list(
    map(
        lambda t: (t[0], ) + analysis_tools.summary_to_record(
            summary_columns_ordered, t[1]), summaries.items()))

models_and_operators = pd.DataFrame.from_records(summary_records,
                                                 columns=['model'] +
                                                 summary_column_names,
                                                 index='model')

print(models_and_operators.to_csv())
