let slideIndex = 1;
showSlides(slideIndex);

var scale = 1,
    panning = false,
    pointX = 0,
    pointY = 0,
    start = {x: 0, y: 0},
    zoom = document.getElementsByTagName('img');

function setTransform(){
    zoom.style.transform = "translate(" + pointX + "px, " + pointY + "px) scale(" + scale + ")";
}

zoom.onmousedown = function(e){
    e.preventDefault();
    start = { x: e.clientX - pointX, y: e.clientY - pointY};
    panning = true;
}

zoom.onmouseup = function(e){
    panning = false;
}

zoom.onmousemove = function(e){
    e.preventDefault;
    if(!panning){
        return;
    }
    pointX = (e.clientX - start.x);
    pointY = (e.clientY - start.y);
    setTransform();
}

zoom.onwheel = function(e){
    e.preventDefault;
    var xs = (e.clientX - pointX) / scale,
        ys = (e.clientY - pointY)/ scale,
        delta = (e.wheelDelta ? e.wheelDelta: -e.deltaY);
    (delta>0) ? (scale *= 1.2) : (scale /= 1.2);
    pointX = e.clientX - xs*scale;
    pointY = e.clientY - ys*scale;

    setTransform();
}


function plusSlides(n){
    showSlides(slideIndex += n);
}

function currentSlide(n){
    showSlides(slideIndex = n);
}

function showSlides(n) {
    let i;
    let slides = document.getElementsByClassName("Slides");
    let dots = document.getElementsByClassName("dot")
    if (n > slides.length) {slideIndex = 1}
    if (n < 1) {slideIndex = slides.length}
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }
    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }
    slides[slideIndex-1].style.display = "block";
    dots[slideIndex-1].className += " active";
}



