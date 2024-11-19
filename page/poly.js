document.addEventListener("DOMContentLoaded", function () {
  stage_poly = new NGL.Stage("viewport-poly");
  stage_poly.setParameters({ backgroundColor: "white" });
  stage_poly
    .loadFile("./page/data/poly/poly-path.gro", {
      defaultRepresentation: true,
      asTrajectory: true,
    })
    .then(function (comp) {
      comp.setName("simulation-poly");
      comp.addRepresentation("ball+stick", {
        type: "protein",
        aspectRatio: 1.7,
        radius: 0.3,
      });
      comp.addTrajectory();
    });

  var toggleSpin_poly = document.getElementById("toggleSpin-poly");
  var isSpinning_poly = false;
  toggleSpin_poly.addEventListener("click", function () {
    if (!isSpinning_poly) {
      stage_poly.setSpin([0, 1, 0], 0.01);
      isSpinning_poly = true;
    } else {
      stage_poly.setSpin(null, null);
      isSpinning_poly = false;
    }
  });

  var toggleRunMDs_poly = document.getElementById("toggleRunMDs-poly");
  var isRunning_poly = false;
  toggleRunMDs_poly.addEventListener("click", function () {
    var trajComp_poly =
      stage_poly.getComponentsByName("simulation-poly").list[0]
        .trajList[0];
    var player_poly = new NGL.TrajectoryPlayer(trajComp_poly.trajectory, {
      step: 60,
      mode: "once",
      interpolateType: "spline",
      timeout: 200,
    });
    if (!isRunning_poly) {
      player_poly.play();
      isRunning_poly = true;
    } else {
      player_poly.play();
      player_poly.pause();
      isRunning_poly = false;
    }
  });

  stage_poly_initial = new NGL.Stage("viewport-poly-initial");
  stage_poly_initial.setParameters({ backgroundColor: "white" });
  stage_poly_initial.loadFile("./page/data/poly/poly-pp2.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      type: "protein",
      aspectRatio: 1.7,
      radius: 0.3,
    });
  });

  var toggleSpin_poly_initial = document.getElementById("toggleSpin-poly-inital");
  var isSpinning_poly_initial = false;
  toggleSpin_poly_initial.addEventListener("click", function () {
    if (!isSpinning_poly_initial) {
      stage_poly_initial.setSpin([0, 1, 0], 0.01);
      isSpinning_poly_initial = true;
    } else {
      stage_poly_initial.setSpin(null, null);
      isSpinning_poly_initial = false;
    }
  });

  stage_poly_target = new NGL.Stage("viewport-poly-target");
  stage_poly_target.setParameters({ backgroundColor: "white" });
  stage_poly_target.loadFile("./page/data/poly/poly-pp1.pdb", {
    defaultRepresentation: true,
    asTrajectory: true,
  }).then(function (comp) {
    comp.addRepresentation("ball+stick", {
      type: "protein",
      aspectRatio: 1.7,
      radius: 0.3,
    });
  });
  
  var toggleSpin_poly_target = document.getElementById("toggleSpin-poly-target")
  var isSpinning_poly_target = false;
  toggleSpin_poly_target.addEventListener("click", function () {
    if (!isSpinning_poly_target) {
      stage_poly_target.setSpin([0, 1, 0], 0.01);
      isSpinning_poly_target = true;
    } else {
      stage_poly_target.setSpin(null, null);
      isSpinning_poly_target = false;
    }
  });
});