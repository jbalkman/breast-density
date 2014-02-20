$(function () {

    var img = null;
    var ih = 1; // image height resized; arbitrarily init
    var canvas = document.getElementById('my-canvas');
    var dropbox = document.getElementById('dropbox');
    var context = canvas.getContext('2d');

    function drawStuff() {
	var iw = canvas.width - 350;
	var s = iw/img.width;

	ih = img.height*s;
	canvas.height = ih;

	context.drawImage(img,(canvas.width-iw)/2,0,iw,ih);
    }  

    $("#processButton").on("click",function(e) {
        e.preventDefault();

	//alert("Curr File: "+curr_file);
	
	//if (curr_files.length > 0) {
	
	//alert("Current file after process button pressed: "+curr_file);
	
	$('#loading-indicator').show();
	$('#dropbox-container').hide();
	$('#results-container').hide();
	
	$.ajax({
	    type: "GET",
	    url: "/process_serve?imgfile="+curr_file,
	    data: {'message':'message'},
	    
	    success: function(resp) {
		if (resp.success > 0) {
		    img = new Image();
		    img.src = "data:image/jpeg;base64," + resp.imagefile;
		    img.onload = function () {
			//alert("Image source from process serve routine: "+img.src);
			resizeCanvas();	    
		    }
		    $("#DENSITY").html(resp.density);
		    $("#DCAT").html(resp.dcat);
		    $("#SIDE").html(resp.side);
		    $("#VIEW").html(resp.view);
		} else {
		    alert("Sorry, there was an error processing the submitted image. Ensure that the mammogram file is of TIFF, JPEG, or DICOM format.");
		}
		$('#loading-indicator').hide();
		$('#dropbox-container').show();
		$('#results-container').show();
		removeImages();
	    }
	});
    }); 
    
    function post_image() {
	
	var message = "message";
	
        alert("In ajax fxn...");
	
	$.ajax({
	    type: "GET",
	    url: "/serve_img?file=example.jpg",
	    data: {'message':'message'},
	    
	    success: function(resp){
		//Get the canvas
		var canvas = document.getElementById('my-canvas');
		var context = canvas.getContext('2d');
		context.fillStyle="#FF0000";
		context.fillRect(0,0,400,400);
		
		// image data
                /*var theImage = new Image();		    
                  var theImage = new Image();		    
                  var bytes = new Uint8Array(resp);
                  var binary = '';
                  for (var i = 0; i < bytes.byteLength; ++i) {
		  binary += String.fromCharCode(bytes[i]);
                  }
                  theImage.src = "data:image/jpeg;base64," + window.btoa(binary);
                  theImage.src = resp;
                  theImage.src = "data:image/jpeg;base64," + resp;
		  alert("Image Src: "+theImage.src);
		  alert("Image Bytes: "+bytes+" Byte Length: "+bytes.byteLength+" Second index: "+String.fromCharCode(bytes[1]));
		  context.drawImage("http://127.0.0.1:5000/serve_img?file=example.jpg",0,0,800,400);*/
	    }
	});
    }

    function removeImages () {
	var dropbox = $('#dropbox'),
	message = $('.message', dropbox);
	$('.preview').remove();
	message.show();
	curr_files = [];
    }
    
    function resizeCanvas() {
        var prev_width = canvas.width;
        var new_width = window.innerWidth;
        //var new_width = window.innerWidth*0.8;

	canvas.width = new_width;
	canvas.height = ih;
	//canvas.height = canvas.height * (new_width/prev_width);	
        drawStuff(); 
    }
   
    // resize the canvas to fill browser window dynamically
    window.addEventListener('resize', resizeCanvas, false); 

});