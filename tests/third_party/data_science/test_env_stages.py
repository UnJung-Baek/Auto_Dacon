import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).parent.parent.parent.parent.resolve())

from third_party.data_science.env_stages import TopDatascienceStage


def test_ramp_stage() -> None:
    has_table_input = True
    has_img_input = False
    has_txt_input = False
    has_table_target = True
    has_regression_target = False
    has_img_target = False
    has_txt_target = False

    top_stage = TopDatascienceStage(
        has_table_input=has_table_input, has_img_input=has_img_input, has_txt_input=has_txt_input,
        has_regression_target=has_regression_target,
        has_table_target=has_table_target, has_img_target=has_img_target, has_txt_target=has_txt_target
    )
    avail_actions = "".join([f"\n\t- {s.name}" for s in top_stage.get_available_stages()])
    print(f"Available actions:{avail_actions}")

    actions = ["Use RAMP solution", "Blend submissions", "Send a submission"]
    for action in actions:
        print(f"Choose: {action}\n")
        top_stage.update(action)
        avail_actions = "".join([f"\n\t- {s.name}" for s in top_stage.get_available_stages()])
        print(f"Available actions:{avail_actions}")

    action = 'Blend submissions'
    print(f"Choose: {action}\n")
    top_stage.update(action)
    avail_actions = "".join([f"\n\t- {s.name}" for s in top_stage.get_available_stages()])
    print(f"Available actions:{avail_actions}")


def test_classical_tab_stage() -> None:
    has_table_input = True
    has_img_input = False
    has_txt_input = False
    has_table_target = True
    has_img_target = False
    has_txt_target = False
    has_regression_target = False

    top_stage = TopDatascienceStage(
        has_table_input=has_table_input, has_img_input=has_img_input, has_txt_input=has_txt_input,
        has_regression_target=has_regression_target,
        has_table_target=has_table_target, has_img_target=has_img_target, has_txt_target=has_txt_target
    )
    avail_actions = "".join([f"\n\t- {s.name}" for s in top_stage.get_available_stages()])
    print(f"Available actions:{avail_actions}")

    actions = ["Use classical ML solutions"]
    add_submission = [
        "Add a classical ML submission", "Add feature engineering",
        # "Add a predictive model", "Suggest hyperparameters", "Train the model"
    ]
    actions += 2 * add_submission
    for action in actions:
        print(f"Choose: {action}\n")
        top_stage.update(action)
        avail_actions = "".join([f"\n\t- {s.name}" for s in top_stage.get_available_stages()])
        print(f"Available actions:{avail_actions}")

    action = 'Submit tab predictions'
    print(f"Choose: {action}\n")
    top_stage.update(action)
    avail_actions = "".join([f"\n\t- {s.name}" for s in top_stage.get_available_stages()])
    print(f"Available actions:{avail_actions}")


def test_dnn_stage(has_regression_target: bool):
    has_table_input = True
    has_img_input = True
    has_txt_input = False
    has_table_target = True
    has_img_target = False
    has_txt_target = False

    top_stage = TopDatascienceStage(
        has_table_input=has_table_input, has_img_input=has_img_input, has_txt_input=has_txt_input,
        has_regression_target=has_regression_target,
        has_table_target=has_table_target, has_img_target=has_img_target, has_txt_target=has_txt_target
    )
    avail_actions = "".join([f"\n\t- {s.name}" for s in top_stage.get_available_stages()])
    print(f"Available actions:{avail_actions}")

    actions = ["Use deep learning-based solutions."]
    add_submission_action = [
        "Add a submission", "Add a deep neural net based submission",
        "Implement tabular embedding",
        "Implement tabular feature engineering",
        "Implement embedding of pre-processed tabular data",
        "Implement image embedding", "Implement image transform", "Implement image modelling",
    ]
    if has_regression_target:
        add_submission_action.append("Implement transform of the tabular regression target")
    add_submission_action.extend([
        "Implement table model head",
        "Add training code",
        "Train submission"
    ])

    actions += add_submission_action * 2

    for action in actions:
        print(f"Choose: {action}\n")
        if action not in avail_actions:
            print(action, avail_actions)
            raise ValueError(f"Action {action} is not available: {avail_actions}")
        top_stage.update(action)
        avail_actions = "".join([f"\n\t- {s.name}" for s in top_stage.get_available_stages()])
        print(f"Available actions:{avail_actions}")

    action = 'Blend existing submissions'
    print(f"Choose: {action}\n")
    top_stage.update(action)
    avail_actions = "".join([f"\n\t- {s.name}" for s in top_stage.get_available_stages()])
    print(f"Available actions:{avail_actions}")


if __name__ == "__main__":
    print("\n-----------\nTest classical tab stages\n-----------\n")
    # test_classical_tab_stage()
    print("\n-----------\nTest dnn stages\n-----------\n")
    test_dnn_stage(has_regression_target=True)
    test_dnn_stage(has_regression_target=False)
