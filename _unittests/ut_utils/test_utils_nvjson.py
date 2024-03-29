"""
@brief      test log(time=5s)
"""
import os
import unittest
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from pyquickhelper.loghelper.buffered_flog import BufferedPrint
from onnxcustom.utils.nvprof2json import (
    convert_trace_to_json, json_to_dataframe,
    json_to_dataframe_streaming)


class TestConvertTraceToJson(ExtTestCase):

    def test_convert_trace_to_json(self):
        """
        This file was generated with the following command line:

        ::

            nvprof -o bench_ortmodule_nn_gpu.sql python plot_orttraining_linear_regression_gpu.py

        To get the profile which can be displayed by the nvidia profiler:

        ::

            nvprof -o bench_ortmodule_nn_gpu.nvvp python plot_orttraining_linear_regression_gpu.py
        """
        temp = get_temp_folder(__file__, 'temp_convert_trace_to_json')
        data = os.path.join(temp, "..", "data",
                            "bench_ortmodule_nn_gpu.sql.zip")
        output = os.path.join(temp, "bench_ortmodule_nn_gpu.json")
        tempf = os.path.join(temp, "bench_ortmodule_nn_gpu.sql")
        buf = BufferedPrint()
        convert_trace_to_json(data, output=output, temporary_file=tempf,
                              verbose=1, fLOG=buf.fprint)
        self.assertIn("step 1 begin.", str(buf))
        jst = convert_trace_to_json(data, temporary_file=tempf)
        self.assertExists(output)
        self.assertExists(tempf)
        df = json_to_dataframe(jst)
        df2 = json_to_dataframe(output)
        with open(output, "r", encoding="utf-8") as f:
            df3 = json_to_dataframe(f)
        self.assertEqual(df.shape, df2.shape)
        self.assertEqual(df.shape, df3.shape)

        self.assertRaise(lambda: json_to_dataframe_streaming(jst, chunksize=100),
                         RuntimeError)
        dfs2 = json_to_dataframe_streaming(output, chunksize=100)
        with open(output, "r", encoding="utf-8") as f:
            dfs3 = json_to_dataframe_streaming(f, chunksize=100)
            shape3 = dfs3.shape
        shape2 = dfs2.shape
        self.assertEqual(shape2, shape3)
        cols = list(df.columns)
        self.assertEqual(cols,
                         ['name', 'ph', 'cat', 'ts',
                          'dur', 'tid', 'pid', 'args', 'ts_sec'])
        self.assertEqual(set(df.ph), {'X'})
        self.assertEqual(set(df.cat), {'cuda'})


if __name__ == "__main__":
    unittest.main()
