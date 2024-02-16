// javascript file that set the functionality of some of the components in the index.html file
// imported to the index.html file via the <script> tag in the end of the body part  


$(document).ready(function() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ scroll back to top button ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // javascript code was taken from https://mdbootstrap.com/docs/standard/extended/back-to-top/
    //Get the button
    let mybutton = document.getElementById("btn-back-to-top");
    // When the user scrolls down 20px from the top of the document, show the button
    window.onscroll = function () {
        scrollFunction();
        };
    function scrollFunction() {
        if (
            document.body.scrollTop > 20 ||
            document.documentElement.scrollTop > 20
        ) {
            mybutton.style.display = "block";
        } else {
            mybutton.style.display = "none";
        }
    }
    // When the user clicks on the button, scroll to the top of the document
    mybutton.addEventListener("click", backToTop);
    function backToTop() {
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
    }
});


// ~~ Stop all other audios
// https://stackoverflow.com/questions/19790506/multiple-audio-html-auto-stop-other-when-current-is-playing-with-javascript
document.addEventListener('play', function(e){
    var audios = document.getElementsByTagName('audio');
    for(var i = 0, len = audios.length; i < len;i++){
        if(audios[i] != e.target){
            audios[i].pause();
        }
    }
}, true);

function toggleCollapseArrow(btnid) {
    a = $('#'+ btnid + ' i')[0];
    if (a.classList.contains('fa-chevron-down')) {
        a.classList.remove('fa-chevron-down');
        a.classList.add('fa-chevron-up');
    } else {
        a.classList.remove('fa-chevron-up');
        a.classList.add('fa-chevron-down');
    }
}

function copyBib() {
    let copyText = $("#citation")[0];
    navigator.clipboard.writeText(copyText.getInnerHTML());
}

function hearMore(btn, from, to) {
    let button = $('#'+btn)[0];
    let table = button.parentElement.parentElement.parentElement;
    for (let i = from; i <= to; i++) {
        table.rows[i].hidden = false;
    }
    button.parentElement.parentElement.hidden = true;
}

// Add a listener for when a toggle div is collapsed back
document.addEventListener('hidden.bs.collapse', function (e) {
    collapsed_div = e.target.id;
    tbody = $('#'+collapsed_div + ' tbody')[0];
    // if there's a show more button, hide all rows from the bottom to the first show more button
    if ($('#'+collapsed_div + ' button').length > 0) {
        btn_id = $('#'+collapsed_div + ' button')[0].id;

        for (let i = tbody.rows.length - 1; i >= 0; i--) {
            if (tbody.rows[i].getElementsByTagName('button').length > 0) {
                if (tbody.rows[i].getElementsByTagName('button')[0].id == btn_id) {
                    tbody.rows[i].hidden = false;
                    break;
                }
            }
            tbody.rows[i].hidden = true;
        }
    }
});