const $ = require('jquery');
const bis_genericio = require('bis_genericio');
const webutil = require('bis_webutil.js');
const bootbox=require('bootbox');
const userPreferences=require('bisweb_userpreferences');
const bisdbase = require('bisweb_dbase');

require('jstree');

/**
 * When loading a file from the server, the user must be able to browse the files. 
 * This class will render a list of files in a window similar to the file system dialog that opens when a user clicks on an <input type='file'> button.
 * 
 * TODO: Back button breaks after adding supplemental files

 TODO:

 Eventually move save filename inside the dialog box -- one dialog box on save
 Have option to apply the filter on not 
 It would be nice to show date and filesize eventually (and sort by size)

*/


class SimpleFileDialog {

    constructor(options = {}) {


        this.separator='/';

        this.mode =  options.mode || 'load',
        this.modalName =  options.modalName || 'File Server Dialog',

        // Key Widgets
        this.modal=null;
        this.okButton=null;
        this.filenameEntry=null;

        
        // This are the callbacks 
        this.fileListFn=null;
        this.fileRequestFn=null;

        // Entries
        this.fileList = null;
        this.currentPath = null;
        this.currentDirectory = null;

        // Filter files
        this.currentFilters=[{ name: 'All Files', extensions: [] }];
        this.newFilters=true;
        this.activeFilterList=this.currentFilters[0].extensions;

        this.previousList=null;
        this.favorites = [];
        this.lastFavorite=null;

    }

    // --------------- GUI Callbacks ------------------------
    /** 
     * Request Filename from GUI
     */
    filenameCallback(name=null) {

        if (this.mode.indexOf('dir')>=0) {
            // Directory Mode
            this.modal.dialog.modal('hide');
            setTimeout( () => {
                this.fileRequestFn(this.currentDirectory);
            },10);
        }
        
        if (name===null) {
            name='';
            try {
                name = this.filenameEntry.val() || '';
            } catch(e) {
                console.log("error",e);
                return;
            }
        }
        
        if (name.length<1) {
            return;
        }

        let outname=this.currentDirectory+this.separator+name;
        
        let sendCallback= (() => {
            this.modal.dialog.modal('hide');
            setTimeout( () => {
                this.fileRequestFn(outname);
            },10);
        });
        
        
        if (this.mode!=='save') {
            return sendCallback();
        }

        // Save Stuff
        bis_genericio.getFileSize(outname).then( () => {
            bootbox.confirm("The file "+outname+" exists. Are you sure you want to overwrite this?", ( (result) => {
                if (result)  {
                    sendCallback();
                }
            }));
        }).catch( () => {
            sendCallback();
        });
    }


    /** Request Directory
     * @param {String} dname -- the name of the directory
     */
    changeDirectory(dname) {
        this.fileListFn( dname,true);
    }

    // --------------- Create GUI ------------------------------
    /**
     * Creates the elements of the file dialog that don't need to be redrawn regularly.
     * 
     * @param {Object} options - Options to specify which elements should be drawn to the dialog 
     */
    
    createDialogUserInterface() {

        this.modal = webutil.createmodal(this.modalName, 'modal-lg');
        $('body').append(this.modal);
        
        this.contentDisplayTemplate = 
        `<div class='col-sm-9 bisweb-file-display'>
            <div><p>Content goes here...</p></div>
        </div>`;

        //make the skeleton for the box
        this.container = $(
            `<div class='container-fluid'>
                <div class='row justify-content-start' style='margin-bottom:10px'>
                   <div class='col-sm-12 bisweb-file-navbar'></div>
                </div>


                <div class='row justify-content-start content-box'>
                    <div class='col-sm-3 favorite-bar'></div>
                    <div class='col-sm-9 bisweb-file-display'>
                      <div class='bisweb-file-list'><p>Content goes here...</p></div>
                    </div>
                </div>
                <div class='row justify-content-start content-box'>
                    <div class='col-sm-3 favorite-buttons'></div>
                    <div class='col-sm-9 bisweb-file-filterbar' style='margin-top:5px'></div>
                </div>

             </div>`);

        
        
        this.createFavorites();
        
        this.okButton = $(`<button type='button' class='btn btn-success'>Load</button>`);
        this.okButton.on('click', (event) => {
            event.preventDefault();
            this.filenameCallback();
        });

        this.modal.footer.append(this.okButton);        
        this.modal.body.append(this.container);        

    }

    /**
     * Create Filters 
     */
    createFilters(filters=null) {

        if (filters) {
            this.currentFilters=JSON.parse(JSON.stringify(filters));
        }  else {
            this.currentFilters=[];
        }
        this.currentFilters.push({ name: 'All Files', extensions: [] });
        this.newFilters=true;
        this.activeFilterList=this.currentFilters[0].extensions;
    }
    
    /**
     * Adds the files specified by list to the file dialog. If the dialog is empty this effectively creates the dialog.
     * The list may also specify extra files fetched by the server, in which case startDirectory will designate the path at which they should be added.
     * 
     * NOTE: The file server that creates the file dialog will provide a few of its functions with the socket bound, e.g. fileListFn, to avoid sharing too many of its internal structures.
     * @param {Array} list - An array of file entries. 
     * @param {String} list.text - The name of the file or folder.
     * @param {String} list.path - The full path indicating where the file is located on the server machine.
     * @param {Object} startDirectory - File entry representing the directory at which the files in list should be added. Undefined means the files represent the files in the current directory
     * @param {Object} rootDirectory - File entry representing the root directory from which startDirectory derives
     * @param {Object} opts - filter options
     */
    openDialog(list, startDirectory = null, rootDirectory='/', opts=null) {

        if (this.modal===null) {
            this.createDialogUserInterface();
        }
        
        this.newFilters=true;
        
        if (opts!==null) {
            opts.filters=opts.filters || null;

            this.createFilters(opts.filters);

            if (opts.title) {
                let newtitle=opts.title;
                if (newtitle) {
                    let title = this.modal.header.find('.modal-title');
                    title.text(newtitle+ ' (using bisweb fileserver)');
                }
            }

            if (opts.mode) 
                this.mode = opts.mode;

            if (this.mode === 'save') {
                this.okButton.text('Save');
                this.displayFiles = true;
            } else if (this.mode.indexOf('dir')>=0) {
                this.okButton.text('Select Directory');
                this.displayFiles = true;
            } else {
                this.okButton.text('Load');
                this.displayFiles = true;
            }
        } else {
            console.log('No opts');
        }

        let initialfilename=null;
        if (opts!==null) {
            if (opts.initialFilename)
                initialfilename=opts.initialFilename;
        }
        
        this.fileList = list;
        this.currentDirectory = startDirectory;


        //keep track of the current directory for the navbar
        this.currentPath = startDirectory;
        this.container.find('.bisweb-file-navbar').empty();


        let filterbar=this.container.find('.bisweb-file-filterbar');
        if (this.currentFilters.length<1 || this.displayFiles===false) {
            filterbar.empty();
            this.activeFilterList=[];
        } else if (this.newFilters===true) {
            filterbar.empty();
            this.newFilters=false;
                
            let filter_label=$("<span>Filter Files: </span>");
            filter_label.css({'padding':'10px'});
            filterbar.append(filter_label);
            let sel=webutil.createselect({
                parent : filterbar,
                values : [],
                callback : (e) => {
                    e.preventDefault();
                    let ind=parseInt(e.target.value);
                    if (ind>=0 && ind<this.currentFilters.length) {
                        this.activeFilterList=this.currentFilters[ind].extensions;
                        this.updateTree(this.previousList,name,rootDirectory);
                    }
                }
            });
            sel.empty();

            let addOption= ((b) => {
                sel.append($(b));
            });

            console.log('adding ', this.currentFilters.join('\n\t'));
            
            for (let i=0;i<this.currentFilters.length;i++) {

                if (this.currentFilters[i].extensions.length>0) {
                    addOption(`<option value="${i}">${this.currentFilters[i].name}, (${this.currentFilters[i].extensions.join(',')})</option>`);
                } else {
                    addOption(`<option value="${i}">${this.currentFilters[i].name}</option>`);
                }
            }
        }

        this.updateTree(list,initialfilename,rootDirectory);

        this.modal.dialog.modal('show');
    }

     /**
     * Adds the files specified by list to the file dialog. If the dialog is empty this effectively creates the dialog.
     * The list may also specify extra files fetched by the server, in which case startDirectory will designate the path at which they should be added.
     * 
     * NOTE: The file server that creates the file dialog will provide a few of its functions with the socket bound, e.g. fileListFn, to avoid sharing too many of its internal structures.
     * @param {Array} list - An array of file entries. May contain children that are themselves file entries. A list entry may contain the following fields:
     * @param {String} list.text - The name of the file or folder.
     * @param {String} list.type - What type of file or folder the entry represents. One of 'picture', 'html', 'js', 'text', 'video', 'audio', 'file', or 'directory'.
     * @param {String} list.path - The full path indicating where the file is located on the server machine.
     * @param {Array} list.children - File entries for each file contained in the list entry. Only for list entries of type 'directory'.
     * @param {Object} startDirectory - File entry representing the directory at which the files in list should be added. Undefined means the files represent the files in the current directory
     */
    createFileList(list, startDirectory = null, opts=null) {

        if (opts!==null) {
            this.filters=opts.suffix || '';
            let newtitle=opts.title || null;
            if (newtitle) {
                let title = this.modal.header.find('.modal-title');
                title.text(newtitle+ ' (using bisweb fileserver)');
            }
        }
        
        //file list is constructed as more files are fetched -- the first request will provide a certain number of files then supplemental requests will build it up.
        //createFileList will be called to make the list from scratch, but should not alter the list after that.
        this.fileList = this.fileList ? this.fileList : list;
        this.currentDirectory = list;

        this.lastDirectories = [];
        this.lastPaths = [];

        //keep track of the current directory for the navbar
        this.currentPath = startDirectory ? startDirectory.path.split('/') : [];
        this.container.find('.file-navbar').empty();

        if (startDirectory) {
            this.expandDirectory(startDirectory.list);
        } else {
            this.expandDirectory(list);
        }   
    }
    
    /**
     * Creates the visual representation of the files specified by list. Called from createFileList (see notes there for format of file entries).
     * Uses jstree to render the list.
     * 
     * Sorts contents before display so that folders are shown first.
     * @param {Array} list - An array of file entries. 
     * @param {String} lastfilename - the last selected filename
     * @param {String} rootDirectory - "the drive" we are looking in

     */
    updateTree(list,lastfilename=null, rootDirectory='/') {

        this.previousList=JSON.parse(JSON.stringify(list));
        
        let fileList = this.container.find('.bisweb-file-list');
        fileList.remove();
        fileList = $(`<div class='bisweb-file-list'></div>`);


        let fileDisplay = this.container.find('.bisweb-file-display');
        fileList.empty();
        fileList.css({ 'max-height' : '300px',
                       'height'     : '300px',
                       'max-width'  : '600px',
                       "overflow-y": "auto",
                       "overflow-x": "auto",
                       "color" : "#0ce3ac",
                       "background-color": "#444444"
                     });


        if (!this.displayFiles) {
            let len=list.length-1;
            for (let i = len; i >=0; i=i-1) {
                if (list[i].type !== 'directory') {
                    list.splice(i, 1);
                    i--;
                } 
            }
        } else if (this.activeFilterList.length>0) {
            console.log('Filtering with',this.activeFilterList);
            let len=list.length-1;
            for (let i = len; i >=0; i=i-1) {
                if (list[i].type !== 'directory') {
                    let ok=this.checkFilenameForFilter(list[i].text,this.activeFilterList);
                    if (!ok) {
                        list.splice(i,1);
                    }
                }
            }
        }
        
        //sort folders ahead of files
        list.sort( (a, b) => {

            let isadir=(a.type === 'directory');
            let isbdir=(b.type === 'directory');

            if (isadir && !isbdir)
                return -1;
            if (isbdir && !isadir)
                return 1;

            let at=a.text.toLowerCase();
            let bt=b.text.toLowerCase();
            
            if (at>bt)
                return 1;
            if (at<bt)
                return -1;
            return 0;
        });

        
        fileList.jstree({
            'core': {
                'data': list,
                'dblclick_toggle': false
            },
            'types': {
                'default': {
                    'icon': 'glyphicon glyphicon-file'
                },
                'file': {
                    'icon': 'glyphicon glyphicon-file'
                },
                'root': {
                    'icon': 'glyphicon glyphicon-home'
                },
                'directory': {
                    'icon': 'glyphicon glyphicon-folder-close'
                },
                'picture': {
                    'icon': 'glyphicon glyphicon-picture'
                },
            },
            'plugins': ["types"]
        });

        //determine based on the type of the node what should happen when the user clicks on it
        $(fileList).on('select_node.jstree', (event, data) => {

            let fname=data.node.original.path;
            if (data.node.type === 'file' || data.node.type ==='picture') {

                let ind=fname.lastIndexOf(this.separator);
                let dname=fname;
                if (ind>0)
                    dname=fname.substr(ind+1,fname.length);
                this.filenameEntry.val(dname);
            } else if ( data.node.type=== 'directory') {
                this.changeDirectory(fname);
            }
        });


        fileDisplay.append(fileList);
        this.updateFileNavbar(lastfilename,rootDirectory);
    }

    /**
     * Updates the list of folders at the top of the file dialog to reflect the folders in the current path.
     * @param {String} lastfilename - the last selected filename
     * @param {String} rootDirectory - "the drive" we are looking in
     */
    updateFileNavbar(lastfilename=null, rootDirectory='/') {

        console.log('root directory', rootDirectory, 'current path', this.currentPath);
        let navbar = this.modal.body.find('.bisweb-file-navbar');
        navbar.empty();
        
        //create navbar buttons for each folder in the current path

        
        let folders=null;
        
        if (rootDirectory.length>1 && this.currentPath.length>=rootDirectory.length) {

            let p=this.currentPath.substr(rootDirectory.length+1,this.currentPath.length);
            let f=p.split(this.separator);
            folders=[ rootDirectory.substr(1,rootDirectory.length)].concat(f);
        } else {
            folders=this.currentPath.split(this.separator);
        }

        
//        console.log('Path=',this.currentPath,'root=',rootDirectory,' folders=',folders.join(', '));
        
        for (let i=folders.length-1;i>=0;i=i-1) {
            if (folders[i].length<1)
                folders.splice(i,1);
        }

        for (let i=-1;i<folders.length;i++) {

            let newPath ='';
            let b="";
            let name="";
            if (i===-1)
                b=`<span class='glyphicon glyphicon-folder-close'></span>`;
            if (i>=0) {
                for (let k=0;k<=i;k++) 
                    newPath=newPath+'/'+folders[k];

                name=folders[i]+'/';
            } else {
                newPath="[Root]";
                name=" [Root]";
            }
            let button = $(`<button type='button' class='btn btn-sm btn-link' style='margin:0px'>${b}${name}</button>`);
            button.on('click', (event) => {
                event.preventDefault();
                this.fileListFn(newPath);
            });
            
            navbar.append(button);
        }

        
        this.filenameEntry=$(`<input type='text'class="btn-link btn-sm" style="width:500px; border-style : dotted; border-color: white">`);
        if (!lastfilename)
            this.filenameEntry.val('');
        else
            this.filenameEntry.val(lastfilename);
        navbar.append(this.filenameEntry);
        this.filenameEntry.keypress( ( (ev) => {
            if ( ev.which == 13 ) {
                this.filenameCallback();
            }
        }));
    }

    addFavorite(pillsBar,elem) {

        let selectPillFromPills = (pill, pills) => {
            for (let otherPill of pills) {
                $(otherPill).removeClass('active');
            }
            $(pill).addClass('active');
        };

        let newPill = $(`<li><a href='#' class="active" style="padding: 2px 2px 2px 2px">${elem.name}</a></li>`);
        this.lastFavorite=elem.path;
        
        newPill.on('click', (event) => {
            event.preventDefault();
            selectPillFromPills(newPill, pillsBar.find('li'));
            this.changeDirectory(elem.path);
            this.lastFavorite=elem.path;
        });
        pillsBar.append(newPill);
    }

    addAllFavorites(pillsBar) {
        pillsBar.empty();
        let l=this.favorites.length;
        for (let i=0;i<l;i++) {
            this.addFavorite(pillsBar,this.favorites[i]);
        }
        this.lastFavorite=null;

    }
    
    createFavorites() {

        let favoriteBar = this.container.find('.favorite-bar');
        let favoriteButtons = this.container.find('.favorite-buttons');

        favoriteBar.css({ 'max-height' : '300px',
                          'height'     : '300px',
                          'max-width'  : '250px',
                          "overflow-y": "auto",
                          "overflow-x": "auto",
                          "color" : "#0ce3ac",
                          "background-color": "#444444"
                    });

        let favoriteButton = $(`<button type='button' class='btn btn-sm btn-link'><span class='glyphicon glyphicon-star-empty'></span>Bookmark</button>`);
        favoriteButtons.append(favoriteButton);
        let favoriteButton2 = $(`<button type='button' class='btn btn-sm btn-link'><span class='glyphicon glyphicon-remove'></span> Remove</button>`);
        favoriteButtons.append(favoriteButton2);

        let pillsHTML = $(`<ul class='nav nav-pills nav-stacked btn-sm'></ul>`);
        favoriteBar.append(pillsHTML);
        
        let pillsBar = favoriteBar.find('.nav.nav-pills');        
        
        //TODO: add folder to localforage ...
        favoriteButton.on('click', (event) => {
            event.preventDefault();
            if (this.favorites.length>8)
                return;
            let name = this.currentDirectory;
            if (name.length>23)
                name='___'+name.substr(name.length-23,name.length);
            let elem = {
                'name' : name,
                'path' : this.currentDirectory,
            };
            if (!this.favorites)
                this.favorites = [];

            let i=0,found=false;
            while (i<this.favorites.length && found===false) {
                if (elem.path===this.favorites[i].path)
                    found=true;
                i=i+1;
            }
            if (!found) {
                this.favorites.push(elem);
                this.addFavorite(pillsBar,elem);
                userPreferences.setItem('favoriteFolders',this.favorites,true);
            }
        });

        favoriteButton2.on('click', (event) => {
            event.preventDefault();
            let i=0;
            while (i<this.favorites.length) {
                if (this.lastFavorite===this.favorites[i].path) {
                    this.favorites.splice(i,1);
                    this.addAllFavorites(pillsBar);
                    userPreferences.setItem('favoriteFolders',this.favorites,true);
                    this.lastFavorite=null;
                    return;
                } 
                i=i+1;
            }
        });
            
        let userPreferencesLoaded = userPreferences.webLoadUserPreferences(bisdbase);
        userPreferencesLoaded.then( () => {
            let f=null;
            try {
                f= userPreferences.getItem('favoriteFolders');
            } catch(e) {
                console.log(e);
            }
            if (f) {
                this.favorites=f;
                this.addAllFavorites(pillsBar);
            }
        }).catch( (e) => {
            console.log('Error',e);
        });
    }
    
    /**
       * @returns {Boolean} if visible return true
       */
    isVisible() {
        let vis=this.modal.dialog.css('display');
        if (vis==='block') {
            return true;
        } 

        return false;
    }
    

    /**
     * Checks a proposed filename against a set of file extension filters to determine whether name should have another kind of filetype applied to it.
     * 
     * @param {String} name - A tentative filename
     * @param {Array} filtersList- A string set of file extensions
     * @returns A properly formatted filename
     */
    fixFilename(name, filterList) {

        for (let i=0;i<filterList.length;i++) {
            let filter=filterList[i];
            let nl=name.length;
            let fl=filter.length;
            if (nl>fl) {
                let subname=name.substr(nl-fl,fl);
                if (subname===filter) {
                    //console.log('Matched filter',filter,subname,' returning',name);
                    return name;
                }
            }
        }
        return name + '.'+ filterList[0];
    }

    checkFilenameForFilter(name,filterList) {

        if (filterList.length<1)
            return true;
        
        for (let i=0;i<filterList.length;i++) {
            let filter=filterList[i];
            let nl=name.length;
            let fl=filter.length;
            if (nl>fl) {
                let subname=name.substr(nl-fl,fl);
                if (subname===filter) {
                    return true;
                }
            }
        }
        return false;
    }
}

module.exports = SimpleFileDialog;
