const $ = require('jquery');
const bis_webutil = require('bis_webutil.js');

class BiswebCardBar extends HTMLElement {

    constructor() {
        super();
        this.cards = [];
    }

    connectedCallback() {
        this.bottomNavbarId = this.getAttribute('bis-botmenubarid');
        bis_webutil.runAfterAllLoaded(() => {
            this.bottomNavbar = document.querySelector(this.bottomNavbarId);
            let navbarTitle = this.getAttribute('bis-cardbartitle');
            this.createBottomNavbar(navbarTitle);
        });
    }

    createBottomNavbar(navbarTitle) {
        let bottomNavElement = $(this.bottomNavbar).find('.navbar.navbar-fixed-bottom');
        let bottomNavElementHeight = bottomNavElement.css('height');

        const cardLayout = $(`
        <nav class='navbar navbar-expand-lg navbar-fixed-bottom navbar-dark' style='min-height: 50px; max-height: 50px;'>
            <div class='pos-f-b'>
                <div id='bisweb-plot-navbar' class='collapse' style='position: absolute; bottom: ${bottomNavElementHeight}'>
                    <div class='bg-dark p-4'>
                        <ul class='nav nav-tabs bisweb-bottom-nav-tabs' role='tablist'>
                            <li class='nav-item'>
                                <a class='nav-link' href='#firstTab' role='tab' data-toggle='tab'>Example</a>
                            </li>
                            <li class='nav-item'> 
                                <a class='nav-link' href='#secondTab' role='tab' data-toggle='tab'>Another example</a>
                            </li>
                        </ul>
                        <div class='tab-content'>
                                <div id='firstTab' class='tab-pane fade' role='tabpanel'>
                                    <a>Hello!</a>
                                </div>
                                <div id='secondTab' class='tab-pane fade' role='tabpanel'>
                                    <a>How's it going?</a>
                                </div>
                        </div>
                    </div>
                </div>
            </div> 
        </nav>
        `);

        const expandButton = $(`
            <button class='btn navbar-toggler' type='button' data-toggle='collapse' data-target='#bisweb-plot-navbar' style='visibility: hidden;'>
            </button>
        `);

        const navbarExpandButton = $(`<span><span class='glyphicon glyphicon-menu-hamburger bisweb-span-button' style='position: relative; top: 3px; left: 5px; margin-right: 10px'></span>${navbarTitle}</span>`);
        navbarExpandButton.on('click', () => { expandButton.click(); });

        //append card layout to the bottom and the expand button to the existing navbar
        bottomNavElement.append(navbarExpandButton);
        bottomNavElement.append(expandButton);
        $(this.bottomNavbar).prepend(cardLayout);
    }
}

bis_webutil.defineElement('bisweb-cardbar', BiswebCardBar);