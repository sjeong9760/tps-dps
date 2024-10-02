document.addEventListener("DOMContentLoaded", function () {
  stage_alanine = new NGL.Stage("viewport-alanine");
  stage_alanine.setParameters({ backgroundColor: "white" });
  stage_alanine
    .loadFile("./page/data/alanine-path.gro", {
      defaultRepresentation: true,
      asTrajectory: true,
    })
    .then(function (comp) {
      comp.setName("simulation-alanine");
      comp.addTrajectory();
    });

  var toggleSpin_alanine = document.getElementById("toggleSpin-alanine");
  var isSpinning_alanine = false;
  toggleSpin_alanine.addEventListener("click", function () {
    if (!isSpinning_alanine) {
      stage_alanine.setSpin([0, 1, 0], 0.01);
      isSpinning_alanine = true;
    } else {
      stage_alanine.setSpin(null, null);
      isSpinning_alanine = false;
    }
  });

  var toggleRunMDs_alanine = document.getElementById("toggleRunMDs-alanine");
  var isRunning_alanine = false;
  toggleRunMDs_alanine.addEventListener("click", function () {
    var trajComp_alanine =
      stage_alanine.getComponentsByName("simulation-alanine").list[0]
        .trajList[0];
    var player_alanine = new NGL.TrajectoryPlayer(trajComp_alanine.trajectory, {
      step: 40,
      mode: "once",
      interpolateType: "spline",
      timeout: 200,
    });
    if (!isRunning_alanine) {
      player_alanine.play();
      isRunning_alanine = true;
    } else {
      player_alanine.play();
      player_alanine.pause();
      isRunning_alanine = false;
    }
  });

  stage_alanine_initial = new NGL.Stage("viewport-alanine-initial");
  stage_alanine_initial.setParameters({ backgroundColor: "white" });
  stage_alanine_initial.loadFile("./page/data/alanine-c5.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  });
  var toggleSpin_alanine_initial = document.getElementById("toggleSpin-alanine-inital");
  var isSpinning_alanine_initial = false;
  toggleSpin_alanine_initial.addEventListener("click", function () {
    if (!isSpinning_alanine_initial) {
      stage_alanine_initial.setSpin([0, 1, 0], 0.01);
      isSpinning_alanine_initial = true;
    } else {
      stage_alanine_initial.setSpin(null, null);
      isSpinning_alanine_initial = false;
    }
  });

  stage_alanine_target = new NGL.Stage("viewport-alanine-target");
  stage_alanine_target.setParameters({ backgroundColor: "white" });
  stage_alanine_target.loadFile("./page/data/alanine-c7ax.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  });
  var toggleSpin_alanine_target = document.getElementById("toggleSpin-alanine-target")
  var isSpinning_alanine_target = false;
  toggleSpin_alanine_target.addEventListener("click", function () {
    if (!isSpinning_alanine_target) {
      stage_alanine_target.setSpin([0, 1, 0], 0.01);
      isSpinning_alanine_target = true;
    } else {
      stage_alanine_target.setSpin(null, null);
      isSpinning_alanine_target = false;
    }
  });
});
