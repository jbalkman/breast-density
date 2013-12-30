$(function () {

	$("#processButton").on("click",function(e) {
            e.preventDefault();
	    
	    $.ajax({
		type: "POST",
		url: "/process?imgfile="+curr_file,
		data: {'message':'message'},

		success: function(resp){
		    var img = new Image();
		    img.src = "/serve_img?file="+resp.file;
		    alert("Image source: "+img.src);
		    img.onload = function () {
			var canvas = document.getElementById("myCanvas");
			var context = canvas.getContext("2d");
			alert("Context: "+context);
			context.drawImage(img, 0, 0, 200, 200);
		    }
		}
	    });


	    /*post_image();    
	    var canvas = document.getElementById('myCanvas');
	    var context = canvas.getContext('2d');
	    //var theImage = document.createElement('img');
	    var theImage = new Image();	
	    theImage.src = "/serve_img?file=2013-12-29-16-50-59-021746.jpg";
	    //alert("Image src: "+theImage.src);
	    context.drawImage(theImage,0,0,200,200);*/
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
		    var canvas = document.getElementById('myCanvas');
		    var context = canvas.getContext('2d');
		    context.fillStyle="#FF0000";
		    context.fillRect(0,0,200,200);

		    // image data
                    var theImage = new Image();		    
                    /*var theImage = new Image();		    
                    var bytes = new Uint8Array(resp);
                    var binary = '';
                    for (var i = 0; i < bytes.byteLength; ++i) {
			binary += String.fromCharCode(bytes[i]);
                    }
                    theImage.src = "data:image/jpeg;base64," + window.btoa(binary);*/
                    //theImage.src = resp;
                    //theImage.src = "data:image/jpeg;base64," + resp;
		    //alert("Image Src: "+theImage.src);
		    //alert("Image Bytes: "+bytes+" Byte Length: "+bytes.byteLength+" Second index: "+String.fromCharCode(bytes[1]));
		    context.drawImage("http://127.0.0.1:5000/serve_img?file=example.jpg",0,0,200,200);
		}
	    });
        };
});