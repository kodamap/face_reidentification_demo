$(function () {

    const detection_cmd = ['async', 'sync', 'face-det', 'face-reid'];

    $.ajaxSetup({ cache: false });
    $('#search-list').fadeOut(500);

    // empty check of label
    // https://stackoverflow.com/questions/17699094/if-input-field-is-empty-disable-submit-button/17699138
    $('.modal-body').find('input').keyup(function () {
        if ($('#face-label').val().length != 0) {
            $('#register').attr('disabled', false);
        } else {
            $('#register').attr('disabled', true);
        }
        if ($('#new-label').val().length != 0) {
            $('#save').attr('disabled', false);
            $('#remove').attr('disabled', true);
        } else {
            $('#save').attr('disabled', true);
            $('#remove').attr('disabled', false);
        }
    });

    // File selector initialize modal 
    function resetFileSelect() {
        $('#register').attr('disabled', true);
        $('#custom-file-label').text('Choose File...');
        $('#face-label').val('');
        $('#preview').remove('');
        $('#customFile').attr('disabled', false);
        $('#custom-file-label').attr('disabled', false);
    };

    $('#modalFaceRegister').on('show.bs.modal', function () {
        resetFileSelect();
    });

    /*
    ref :https://www.w3schools.com/bootstrap4/bootstrap_forms_custom.asp
         https://cccabinet.jpn.org/bootstrap4/javascript/forms/file-browser
    */
    // Add the following code if you want the name of the file appear on select
    $(".custom-file-input").on("change", function (e) {
        let files = e.target.files;
        let filename = files[0].name;
        let filetype = files[0].type;

        if (files && files[0]) {
            var reader = new FileReader();

            $(this).parents('.input-group').after('<div id="preview"></div>');

            reader.onload = (function (e) {
                if (filetype.match('image.*')) {
                    var $html = ['<div class="d-inline-block mr-1 mt-1"><img class="img-thumbnail" src="', e.target.result, '" title="', escape(filename), '" style="height:100px;" /><div class="small text-muted text-center">', escape(filename), '</div></div>'].join('');
                } else {
                    var $html = ['<div class="small text-muted text-center">', 'Invalid file type:' + filetype, '</div>'].join('');
                }
                $('#preview').append($html);
            });
            reader.readAsDataURL(files[0]);
            $('#customFile').attr('disabled', true);
            $('#custom-file-label').attr('disabled', true);
        }

        $(this).next('.custom-file-label').html(+ files.length + 'selected');

        $('#reset').click(function () {
            resetFileSelect();
        });

        // post register action
        $('#register').on('click', function () {
            let command = JSON.stringify({
                "command": "register", "label": $('#face-label').val(), "data": reader.result
            });
            post('/registrar', command);
        });
    });

    function reloadFaceList() {
        $('#face-list').fadeOut(1000);
        $('#face-list').load('/ #face-list');
        $('#face-list').fadeIn(1000);
        $('#video_feed').slideDown(200);
    }

    // post register action in modal
    $('#modalFacePreview').on('show.bs.modal', function (event) {
        let button = $(event.relatedTarget);
        let label = button.data('label');
        let face = button.data('face');

        // initialize modal 
        $('#save').attr('disabled', true);
        $('#new-label').val('');

        // put data on modal 
        $('.modal-body').children('img').attr('src', face)
        $('.modal-body').children('h5').text(label)

        $('#remove').off('click').on('click', function () {
            let command = JSON.stringify({ "command": "remove", "label": label });
            post('/registrar', command);

        });
        $('#save').off('click').on('click', function () {
            let newlabel = $('#new-label').val();
            let command = JSON.stringify({ "command": "save", "label": label, "newlabel": newlabel });
            post('/registrar', command);

        });
    });

    // flip frame
    $('#flip').on('click', function () {
        let command = JSON.stringify({ "command": "flip" });
        post('/flip', command);
    });

    // caputure action
    $('#capture').on('click', function () {
        let command = JSON.stringify({ "command": "capture" });
        post('/registrar', command);
    });

    // reload face list
    $('#reload').on('click', function () {
        let command = JSON.stringify({ "command": "reload" });
        post('/registrar', command);
        reloadFaceList();
    });


    // search face
    //$("label[id*='search-btn']").on('click', function () {
    $('#face-list').on('click', '.btn', function () {
        let label = $(this).find('input').data('label');
        let command = JSON.stringify({ "command": "search", "label": label });
        post('/search', command);
    });

    // post detection action
    $('.btn').on('click', function (e) {

        var command = JSON.stringify({ "command": $('#' + $(this).attr('id')).val() });

        if (JSON.parse(command).command == "") {
            var command = JSON.stringify({ "command": $(this).find('input').val() });
        }

        //console.log("btn", command)

        if (detection_cmd.includes(JSON.parse(command).command)) {
            post('/detection', command);
        }

    });

    // ajax post
    function post(url, command) {
        $.ajax({
            type: 'POST',
            url: url,
            data: command,
            contentType: 'application/json',
            timeout: 10000
        }).done(function (data, textStatus) {
            let post_command = JSON.parse(command).command;
            let is_async = JSON.parse(data.ResultSet).is_async;
            let flip_code = JSON.parse(data.ResultSet).flip_code;
            let is_fd = JSON.parse(data.ResultSet).is_fd;
            let is_fi = JSON.parse(data.ResultSet).is_fi;

            //console.log("post_command", post_command);

            //$("#res").text("command: " + post_command + " async: " + is_async + " flip: " + flip_code + " face: " + is_fd + " reid: " + is_fi + " Status: " + textStatus);
            $("#res").text("Command: " + post_command + " Status: " + textStatus);
            $('#search-list').fadeOut(100);

            if (JSON.parse(command).command == 'async') {
                $("#async").attr('class', 'btn btn-danger');
                $("#sync").attr('class', 'btn btn-dark');
            }

            if (JSON.parse(command).command == 'sync') {
                $("#sync").attr('class', 'btn btn-danger');
                $("#async").attr('class', 'btn btn-dark');
            }

            if (is_fd && post_command == "face-det") {
                $('#video_feed').fadeIn(100);
                $("#face-det").attr('class', 'btn btn-secondary active');
                $("#face-reid").attr('class', 'btn btn-secondary');
            } else if (!is_fd) {
                $("#face-det").attr('class', 'btn btn-secondary');
            }

            if (is_fi && post_command == "face-reid") {
                $('#video_feed').fadeIn(100);
                $("#face-reid").attr('class', 'btn btn-secondary active');
                $("#face-det").attr('class', 'btn btn-secondary');
            } else if (!is_fi) {
                $("#face-reid").attr('class', 'btn btn-secondary');
            }

            if (JSON.parse(command).command == 'register') {
                reloadFaceList();
            }

            if (JSON.parse(command).command == 'capture') {
                reloadFaceList();
            }

            if (JSON.parse(command).command == 'remove') {
                $('#' + JSON.parse(command).label + "-search-btn").fadeOut(1000);
                $('#' + JSON.parse(command).label + "-label").fadeOut(1000);
                $('#video_feed').slideDown(200);

            }

            if (JSON.parse(command).command == 'save') {
                reloadFaceList();
            }

            if (JSON.parse(command).command == 'search') {
                $('#video_feed').slideUp(200);
                $('#search-list').load('/ #search-list');
                $('#search-list').slideDown(200);
            }

        }).fail(function (jqXHR, textStatus, errorThrown) {
            $("#res").text(textStatus + ":" + jqXHR.status + " " + errorThrown);
        });
        return false;
    }
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });
});

