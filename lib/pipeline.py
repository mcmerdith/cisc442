from dataclasses import dataclass
from lib.executable import Compare, Executable, LoadImage, SaveImage, build_operation


@dataclass(init=False, kw_only=True)
class Pipeline(Executable):
    pipeline: list[Executable]

    def __init__(self, *, pipeline: list[dict] = None, index: int, **kwargs):
        super().__init__()
        self._id = str(index)
        # data validation
        if pipeline and "operation" in kwargs:
            raise ValueError(
                f"Cannot specify both pipeline and operation: {kwargs}"
            )

        if not "source" in kwargs:
            raise ValueError(f"Missing source: {kwargs}")

        if pipeline:
            # build the pipeline
            self.pipeline = [build_operation(
                op, id=f"{index}.{i}") for i, op in enumerate(pipeline)]

        elif "operation" in kwargs:
            # build a pipelinee from a single operation
            self.pipeline = [build_operation(kwargs, id=f"{index}")]

        else:
            raise ValueError(f"Illegal pipeline (no operations): {kwargs}")

        # add the loading and saving steps
        self.pipeline.insert(0, LoadImage(
            name=kwargs["source"], _id=f"{index}"
        ))

        # add the saving step
        if "output" in kwargs:
            self.pipeline.append(
                SaveImage(name=kwargs["output"], _id=f"{index}")
            )

        # add the comparison step
        if "test" in kwargs:
            self.pipeline.append(
                Compare(reference=kwargs["test"],
                        test=True, _id=f"{index}:TEST")
            )

    def execute(self):
        super().execute()
        # stitch operations together into a pipeline
        output = None
        for op in self.pipeline:
            output = op.pipe_in(output).execute()
        return output


def build_executor(steps: list[dict] | dict):
    if isinstance(steps, dict):
        return Pipeline(**steps)

    steps = [Pipeline(**step, index=i) for i, step in enumerate(steps)]

    return steps
