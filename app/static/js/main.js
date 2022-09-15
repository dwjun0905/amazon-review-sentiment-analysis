$(document).ready(function () {
    // Init
    $('.loader').hide();

    // Predict
    $('#btn-predict').click(function () {
        // var form_data = new FormData($('#inputUrl')[0]);

        // Show loading animation
        // $(this).hide();
        $('.loader').show();

        // $.ajax({
        //     type: 'POST',
        //     url: '/prediction',
        //     data: form_data,
        //     contentType: false,
        //     cache: false,
        //     processData: false,
        //     async: true,
        //     success: function (data) {
        //         // Get and display the result
        //         console.log('Success!');
        //     },
        // });
    });

});


