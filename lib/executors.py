from dataclasses import dataclass
from lib.operations import Executable, build_operation


@dataclass(init=False, kw_only=True)
class Pipeline(Executable):
    pipeline: list[Executable]
    source: str
    output: str

    def __init__(self, *, pipeline: list[dict] = None, **kwargs):
        # data validation
        if pipeline and "operation" in kwargs:
            raise ValueError("Cannot specify both pipeline and operation")

        if not "source" in kwargs or not "output" in kwargs:
            raise ValueError("Missing source or output")

        # set the source and output
        if pipeline:
            # build the pipeline
            self.pipeline = [build_operation(op) for op in pipeline]

        elif "operation" in kwargs:
            # build a pipelinee from a single operation
            self.pipeline = [build_operation(kwargs)]

        else:
            raise ValueError(f"Illegal pipeline (no operations): {kwargs}")

        # add the loading and saving steps
        self.pipeline.insert(0, build_operation(
            {"operation": "loadImage", "name": kwargs["source"]}
        ))
        self.pipeline.append(build_operation(
            {"operation": "saveImage", "name": kwargs["output"]}
        ))

    def pipe(self, image):
        # do nothing
        return super().pipe(image)

    def execute(self):
        # stitch operations together into a pipeline
        output = None
        for op in self.pipeline:
            output = op.pipe(output).execute()


def build_executor(steps: list[dict] | dict):
    if isinstance(steps, dict):
        return Pipeline(**steps)

    steps = [Pipeline(**step) for step in steps]

    return steps
